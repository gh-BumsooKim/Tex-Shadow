import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F
import trimesh
import rembg

from cam_utils import orbit_camera, OrbitCamera
from mesh_renderer import Renderer
from mesh import Mesh, safe_normalize

from grid_put import mipmap_linear_grid_put_2d

def cal_distance_map(p1, p2):
    distance = p1 - p2
    
    distance_pos = torch.clamp(distance, min=0.0)
    distance_neg = torch.clamp(-distance, min=0.0)
    
    dd = distance_pos + distance_neg
    dd = dd.sum(dim=-1, keepdim=True) / 3

    return distance_pos, distance_neg, dd

def realign_mesh(mesh, path=None, renderer=None, radius=3.8, cam=None, texture_size=1024, device='cuda'):
    # perform texture extraction
    print(f"[INFO] unwrap uv...")
    h = w = texture_size
    mesh.auto_uv()
    mesh.auto_normal()

    albedo = torch.zeros((h, w, 3), device=device, dtype=torch.float32)
    cnt = torch.zeros((h, w, 1), device=device, dtype=torch.float32)

    vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
    hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

    render_resolution = 512

    import nvdiffrast.torch as dr

    glctx = dr.RasterizeGLContext()

    for ver, hor in zip(vers, hors):
        pose        = orbit_camera(ver, hor, radius)
        fixed_cam   = pose, cam.perspective
        
        ref_size = 512
        ssaa     = 1
        cur_out = renderer.render(*fixed_cam, ref_size, ref_size, ssaa=ssaa)

        # [H, W, 3] to [1, 3, H, W] 
        rgbs = cur_out["image"].unsqueeze(0).permute(0, 3, 1, 2) # in [0, 1]
 
        # get coordinate in texture image
        pose = torch.from_numpy(pose.astype(np.float32)).to(device)
        proj = torch.from_numpy(cam.perspective.astype(np.float32)).to(device)

        v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T
        rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
        depth = depth.squeeze(0) # [H, W, 1]

        alpha = (rast[0, ..., 3:] > 0).float()

        uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

        normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]
        viewcos = rot_normal[..., [2]]

        mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
        mask = mask.view(-1)

        uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
        rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
        
        # update texture image
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
            h, w,
            uvs[..., [1, 0]] * 2 - 1,
            rgbs,
            min_resolution=256,
            return_count=True,
        )
        mask = cnt.squeeze(-1) < 0.1
        albedo[mask] += cur_albedo[mask]
        cnt[mask] += cur_cnt[mask]

    mask = cnt.squeeze(-1) > 0
    albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

    mask = mask.view(h, w)

    albedo = albedo.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    # dilate texture
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    inpaint_region = binary_dilation(mask, iterations=32)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        search_coords
    )
    _, indices = knn.kneighbors(inpaint_coords)

    albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

    mesh.albedo = torch.from_numpy(albedo).to(device)
    
    path = path.replace('swap_mesh', 'realign_mesh')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    mesh.write(path)

    print(f"[INFO] save model to {path}.")
    
    return path 

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_zero123 = None

        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(opt).to(self.device)
        self.renderer.get_clean_mesh()
        self.renderer.unwarp_uv(cam=self.cam)

        # input image
        self.input_img = []
        self.input_mask = []
        self.input_img_torch = []
        self.input_mask_torch = []
        self.input_img_torch_channel_last  = []
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        self.sil_init = None

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        if self.opt.train_geo_sep:
            params = self.renderer.get_params()
            self.optimizer = torch.optim.Adam(params[0])
            self.optimizer_geo = torch.optim.Adam(params[1])
        else:
            self.optimizer = torch.optim.Adam(self.renderer.get_params())

        self.fixed_cam = [(orbit_camera(0,  0, self.opt.radius), self.cam.perspective),
                          (orbit_camera(0, 90, self.opt.radius), self.cam.perspective)] 
        
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")
            
        # input image
        if self.input_img is not None:
            for input_img, input_mask in zip(self.input_img, self.input_mask):
                input_img_torch = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                input_img_torch = F.interpolate(input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

                input_mask_torch = torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
                input_mask_torch = F.interpolate(input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
                input_img_torch_channel_last = input_img_torch[0].permute(1,2,0).contiguous()
                
                self.input_img_torch.append(input_img_torch)
                self.input_mask_torch.append(input_mask_torch)
                self.input_img_torch_channel_last.append(input_img_torch_channel_last)

        # prepare embeddings
        with torch.no_grad():
            if self.enable_zero123:
               self.guidance_zero123.get_img_embeds(self.input_img_torch[0])

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        
        for _ in range(self.train_steps):
            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters_refine)

            loss = 0

            ### known view
            if self.input_img_torch is not None and not self.opt.imagedream:
                ssaa = 1
                
                for r_idx, (fixed_cam, input_img_torch_channel_last, input_mask) in enumerate(zip(self.fixed_cam, self.input_img_torch_channel_last, self.input_mask_torch)):
                    out = self.renderer.render(*fixed_cam, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)

                    # rgb loss
                    image = out["image"] # [H, W, 3] in [0, 1]
                    loss = loss + 4 * F.mse_loss(image * input_mask[0][0][:,:,None], input_img_torch_channel_last * input_mask[0][0][:,:,None])
            
            ### novel view (manual batch)
            render_resolution = 512
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)
            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                # random render resolution
                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                image = out["image"] # [H, W, 3] in [0, 1]
                image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

                images.append(image)

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            # guidance loss
            strength = step_ratio * 0.15 + 0.8

            if self.enable_zero123 and self.step > 400:
                refined_images = self.guidance_zero123.refine(images, vers, hors, radii, strength=strength, default_elevation=self.opt.elevation).float()
                refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                loss = loss + self.opt.lambda_zero123 * F.mse_loss(images, refined_images)

            # optimize step
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )
 
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        
        import glob 
        img_list = sorted(glob.glob(os.path.join(file, '*.*')))
        
        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img.shape[-1] == 3:
                if self.bg_remover is None:
                    self.bg_remover = rembg.new_session()
                img = rembg.remove(img, session=self.bg_remover)

            img = cv2.resize(
                img, (self.W, self.H), interpolation=cv2.INTER_AREA
            )
            img = img.astype(np.float32) / 255.0

            input_mask = img[..., 3:]
            # white bg
            input_img = img[..., :3] * input_mask + (
                1 - input_mask
            )
            # bgr to rgb
            input_img = input_img[..., ::-1].copy()
            
            self.input_mask.append(input_mask)
            self.input_img.append(input_img)
    
    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.opt.save_path + '.' + self.opt.mesh_format)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
 
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
        self.save_model()
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # auto find mesh from stage 1
    if opt.mesh is None:
        default_path = os.path.join(opt.outdir, opt.save_path + '_mesh.' + opt.mesh_format)
        if os.path.exists(default_path):
            opt.mesh = default_path
        else:
            raise ValueError(f"Cannot find mesh from {default_path}, must specify --mesh explicitly!")

    gui = GUI(opt)
 
    gui.train(opt.iters_refine)