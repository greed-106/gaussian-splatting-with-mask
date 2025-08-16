import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Sparse Adam requested but not installed. Install with: pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            image *= viewpoint_cam.alpha_mask.cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.original_mask.cuda()

        Ll1 = l1_loss(image * mask, gt_image * mask)
        ssim_value = ssim(image * mask, gt_image * mask)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Adaptive weight scheduling based on training progress
        if hasattr(opt, 'adaptive_weights') and opt.adaptive_weights:
            progress = iteration / opt.iterations
            # Gradually increase perceptual and edge loss weights
            adaptive_perceptual_weight = opt.lambda_perceptual * min(1.0, progress * 2.0) if hasattr(opt, 'lambda_perceptual') else 0
            adaptive_edge_weight = opt.lambda_edge * min(1.0, progress * 3.0) if hasattr(opt, 'lambda_edge') else 0
        else:
            adaptive_perceptual_weight = getattr(opt, 'lambda_perceptual', 0)
            adaptive_edge_weight = getattr(opt, 'lambda_edge', 0)

        # Add perceptual loss for better visual quality
        Lperceptual = 0
        if adaptive_perceptual_weight > 0 and iteration > 1000:  # Only start after some iterations
            try:
                # Simple perceptual loss using L2 on downsampled images
                image_down = F.interpolate(image.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)
                gt_down = F.interpolate(gt_image.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)
                mask_down = F.interpolate(mask.unsqueeze(0), scale_factor=0.5, mode='nearest')
                
                Lperceptual = l2_loss(image_down * mask_down, gt_down * mask_down)
                if torch.isfinite(Lperceptual) and Lperceptual > 0:
                    loss += adaptive_perceptual_weight * Lperceptual
                else:
                    Lperceptual = 0
            except Exception as e:
                Lperceptual = 0

        # Add edge-aware loss for better detail preservation
        Ledge = 0
        if adaptive_edge_weight > 0 and iteration > 2000:  # Start later
            try:
                # Sobel edge detection
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
                
                # Apply edge detection to each channel
                edge_image = torch.zeros_like(image)
                edge_gt = torch.zeros_like(gt_image)
                
                for c in range(image.shape[0]):
                    img_c = image[c:c+1].unsqueeze(0)
                    gt_c = gt_image[c:c+1].unsqueeze(0)
                    
                    edge_x = F.conv2d(img_c, sobel_x, padding=1)
                    edge_y = F.conv2d(img_c, sobel_y, padding=1)
                    edge_image[c] = torch.sqrt(edge_x.squeeze()**2 + edge_y.squeeze()**2)
                    
                    edge_x_gt = F.conv2d(gt_c, sobel_x, padding=1)
                    edge_y_gt = F.conv2d(gt_c, sobel_y, padding=1)
                    edge_gt[c] = torch.sqrt(edge_x_gt.squeeze()**2 + edge_y_gt.squeeze()**2)
                
                Ledge = l1_loss(edge_image * mask, edge_gt * mask)
                if torch.isfinite(Ledge) and Ledge > 0:
                    loss += adaptive_edge_weight * Ledge
                else:
                    Ledge = 0
            except Exception as e:
                Ledge = 0

        Ll1depth = 0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth

        # Add color consistency loss for better color reproduction
        Lcolor = 0
        if hasattr(opt, 'lambda_color') and opt.lambda_color > 0 and iteration > 3000:  # Start even later
            try:
                # Color histogram matching loss
                def compute_histogram_loss(img1, img2, mask, bins=64):
                    # Compute color histograms for each channel
                    hist_loss = 0
                    for c in range(img1.shape[0]):
                        # Handle different mask dimensions
                        if mask.dim() == 3 and mask.shape[0] == img1.shape[0]:
                            mask_c = mask[c]
                        elif mask.dim() == 2:
                            mask_c = mask
                        else:
                            mask_c = mask[0] if mask.dim() == 3 else mask
                        
                        img1_c = img1[c] * mask_c
                        img2_c = img2[c] * mask_c
                        
                        # Compute histograms
                        hist1 = torch.histc(img1_c, bins=bins, min=0, max=1)
                        hist2 = torch.histc(img2_c, bins=bins, min=0, max=1)
                        
                        # Normalize histograms
                        hist1 = hist1 / (hist1.sum() + 1e-8)
                        hist2 = hist2 / (hist2.sum() + 1e-8)
                        
                        # Compute histogram distance
                        hist_loss += torch.abs(hist1 - hist2).mean()
                    
                    return hist_loss / img1.shape[0]
                
                Lcolor = compute_histogram_loss(image, gt_image, mask)
                if torch.isfinite(Lcolor) and Lcolor > 0:
                    loss += opt.lambda_color * Lcolor
                else:
                    Lcolor = 0
            except Exception as e:
                Lcolor = 0

        # Add gradient loss for better detail preservation
        Lgradient = 0
        if hasattr(opt, 'lambda_gradient') and opt.lambda_gradient > 0 and iteration > 4000:  # Start latest
            try:
                # Compute gradients using Sobel filters
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
                
                grad_loss = 0
                for c in range(image.shape[0]):
                    img_c = image[c:c+1].unsqueeze(0)
                    gt_c = gt_image[c:c+1].unsqueeze(0)
                    mask_c = mask[c:c+1].unsqueeze(0) if mask.dim() == 3 else mask.unsqueeze(0).unsqueeze(0)
                    
                    # Compute gradients
                    grad_x_img = F.conv2d(img_c, sobel_x, padding=1)
                    grad_y_img = F.conv2d(img_c, sobel_y, padding=1)
                    grad_x_gt = F.conv2d(gt_c, sobel_x, padding=1)
                    grad_y_gt = F.conv2d(gt_c, sobel_y, padding=1)
                    
                    # Gradient magnitude
                    grad_mag_img = torch.sqrt(grad_x_img**2 + grad_y_img**2)
                    grad_mag_gt = torch.sqrt(grad_x_gt**2 + grad_y_gt**2)
                    
                    grad_loss += l1_loss(grad_mag_img * mask_c, grad_mag_gt * mask_c)
                
                Lgradient = grad_loss / image.shape[0]
                if torch.isfinite(Lgradient) and Lgradient > 0:
                    loss += opt.lambda_gradient * Lgradient
                else:
                    Lgradient = 0
            except Exception as e:
                Lgradient = 0

        # Add regularization loss for better training stability
        Lreg = 0
        if hasattr(opt, 'lambda_reg') and opt.lambda_reg > 0 and iteration > 5000:  # Start very late
            try:
                # Opacity regularization to prevent overfitting
                # Check if gaussians has opacity attribute or method
                if hasattr(gaussians, '_opacity'):
                    opacity = gaussians._opacity
                    # Encourage reasonable opacity values (not too close to 0 or 1)
                    opacity_reg = torch.mean(torch.abs(opacity - 0.5))
                    Lreg = opt.lambda_reg * opacity_reg
                    if torch.isfinite(Lreg) and Lreg > 0:
                        loss += Lreg
                    else:
                        Lreg = 0
                elif hasattr(gaussians, 'get_opacity'):
                    opacity = gaussians.get_opacity()
                    if isinstance(opacity, torch.Tensor):
                        opacity_reg = torch.mean(torch.abs(opacity - 0.5))
                        Lreg = opt.lambda_reg * opacity_reg
                        if torch.isfinite(Lreg) and Lreg > 0:
                            loss += Lreg
                        else:
                            Lreg = 0
            except Exception as e:
                Lreg = 0

        # Final loss stability check
        if not torch.isfinite(loss):
            print(f"Warning: Loss became non-finite at iteration {iteration}, resetting to L1 loss only")
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            Lperceptual = 0
            Ledge = 0
            Lcolor = 0
            Lgradient = 0
            Lreg = 0

        loss.backward()

        with torch.no_grad():
            # Early pruning of foreground points based on projection mask
            if iteration == 100:
                mask_2d = viewpoint_cam.original_mask[0]
                proj_xy = render_pkg["viewspace_points"]
                H, W = mask_2d.shape[-2:]
                proj_x = (proj_xy[:, 0] * W).long().clamp(0, W-1)
                proj_y = (proj_xy[:, 1] * H).long().clamp(0, H-1)
                keep = (mask_2d[proj_y, proj_x] > 0.5)
                print(f"[DEBUG] mask_2d.sum(): {mask_2d.sum().item()} / {H*W}")
                print(f"[DEBUG] proj_xy.shape: {proj_xy.shape}")
                print(f"[DEBUG] Sampled mask values: {mask_2d[proj_y, proj_x][:10]}")
                print(f"[Prune] Keeping {keep.sum()} / {keep.shape[0]} Gaussians (background only)")

                gaussians.prune_by_mask(keep)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Depth Loss": f"{ema_Ll1depth_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, 0.0, testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[:10])
    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))
    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss/total_loss', loss.item(), iteration)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 7000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing", args.model_path)
    safe_state(args.quiet)
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")
