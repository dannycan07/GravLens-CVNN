"""
Generate synthetic galaxy-galaxy strong-lensing images (SIE lens + Sersic
source + Sersic lens light), following the setup of Ancona-Flores et al.
"Enhancing Gravitational Lens Study with Deep Learning" (arXiv:2603.06339).

Each image is rendered noise-free at a fixed pixel grid, and labeled with the
four SIE parameters the paper predicts: Einstein radius (theta_E), axis ratio
(f), and the two ellipticity components (eps_x, eps_y), related to (f, phi) by
eps_x = (1-f)/(1+f)*cos(2*phi),  eps_y = (1-f)/(1+f)*sin(2*phi)
(lenstronomy's own SIE ellipticity convention (e1, e2) uses this identical
definition, so e1/e2 in the SIE kwargs ARE eps_x/eps_y directly.)

Usage:
    python generate_lens_data.py --n 6000 --out training_data/lens
"""

import argparse
import os

import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import param_util, util

NPIX = 64
DELTA_PIX = 0.06  # arcsec/pixel

# Parameter ranges. theta_E is capped well under the field-of-view radius
# (NPIX*DELTA_PIX/2 = 1.92") so the Einstein ring stays visible in-frame, and
# sampled from a right-skewed distribution (small radii far more common than
# large ones) matching the paper's reported theta_E distribution (Fig. 3).
THETA_E_MIN, THETA_E_MAX = 0.15, 1.6
F_RANGE = (0.3, 1.0)


def build_image_model():
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(
        num_pix=NPIX, delta_pix=DELTA_PIX, subgrid_res=1
    )
    dummy = np.zeros((NPIX, NPIX))
    data = ImageData(dummy, ra_at_xy_0=ra_at_xy_0, dec_at_xy_0=dec_at_xy_0, transform_pix2angle=Mpix2coord)
    psf = PSF(psf_type="NONE")

    lens_model = LensModel(["SIE"])
    source_model = LightModel(["SERSIC_ELLIPSE"])
    lens_light_model = LightModel(["SERSIC"])

    return ImageModel(data, psf, lens_model, source_model, lens_light_model)


def sample_system(rng):
    theta_E = np.clip(rng.gamma(shape=2.0, scale=0.35) + THETA_E_MIN, THETA_E_MIN, THETA_E_MAX)
    f = rng.uniform(*F_RANGE)
    phi = rng.uniform(0, np.pi)
    eps_x, eps_y = param_util.phi_q2_ellipticity(phi, f)

    kwargs_lens = [{
        "theta_E": theta_E, "e1": eps_x, "e2": eps_y,
        "center_x": 0.0, "center_y": 0.0,
    }]

    # Random source: position, size, shape, brightness (mirrors the paper's
    # source space parameters Theta_s = (Re, eps_sx, eps_sy, x, y))
    src_phi = rng.uniform(0, np.pi)
    src_q = rng.uniform(0.5, 1.0)
    src_e1, src_e2 = param_util.phi_q2_ellipticity(src_phi, src_q)
    kwargs_source = [{
        "amp": rng.uniform(5, 15),
        "R_sersic": rng.uniform(0.1, 0.4),
        "n_sersic": 2.0,
        "e1": src_e1, "e2": src_e2,
        "center_x": rng.uniform(-0.15, 0.15),
        "center_y": rng.uniform(-0.15, 0.15),
    }]

    kwargs_lens_light = [{
        "amp": rng.uniform(3, 8),
        "R_sersic": rng.uniform(0.3, 0.7),
        "n_sersic": 2.0,
        "center_x": 0.0, "center_y": 0.0,
    }]

    return kwargs_lens, kwargs_source, kwargs_lens_light, (theta_E, f, eps_x, eps_y)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SIE lens images")
    parser.add_argument("--n", type=int, default=6000, help="number of images to generate")
    parser.add_argument("--out", type=str, default="training_data/lens", help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    image_model = build_image_model()

    images = np.zeros((args.n, NPIX, NPIX), dtype=np.float32)
    labels = np.zeros((args.n, 4), dtype=np.float32)  # theta_E, f, eps_x, eps_y

    for i in range(args.n):
        kwargs_lens, kwargs_source, kwargs_lens_light, params = sample_system(rng)
        img = image_model.image(kwargs_lens, kwargs_source, kwargs_lens_light)
        images[i] = img.astype(np.float32)
        labels[i] = params

        if (i + 1) % max(1, args.n // 20) == 0:
            print(f"  {i + 1}/{args.n} done")

    np.save(os.path.join(args.out, "images.npy"), images)
    np.save(os.path.join(args.out, "labels.npy"), labels)
    print(f"Saved {args.n} images to {args.out} (images.npy shape {images.shape}, labels.npy shape {labels.shape})")
    print("Label columns: [theta_E, f, eps_x, eps_y]")


if __name__ == "__main__":
    main()
