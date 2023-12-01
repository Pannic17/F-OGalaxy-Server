import time

import cv2
import numpy as np
import torch
from PIL import Image
from RealESRGAN import RealESRGAN
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler, \
    StableDiffusionInpaintPipeline
from matplotlib import pyplot as plt


def image_canny(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def display_image(image):
    plt.imshow(image)
    plt.axis('off')  # No axes for this image
    plt.show()


def save_image(image, prefix):
    path = prefix.format(int(time.time()))
    image.save(path)
    return path


def cv_add_vignette(image, width, height, center_fraction=0.9):
    x = np.arange(width, dtype=np.float32) - width / 2
    y = np.arange(height, dtype=np.float32) - height / 2
    x, y = np.meshgrid(x, y)
    distance_from_center = np.sqrt(x ** 2 + y ** 2)

    center_radius = min(height, width) / 2 * center_fraction
    radius = min(height, width) / 2
    vignette_mask = np.clip(1 - (distance_from_center - center_radius) / (radius - center_radius), 0, 1)
    vignette_mask = cv2.merge([vignette_mask, vignette_mask, vignette_mask])
    vignette_image = image * vignette_mask
    return vignette_image


def cv_create_vignette_mask(width, height, center_fraction=0.8):
    center_radius = min(height, width) / 2 * center_fraction
    vignette_center = np.ones((height, width, 3), dtype=np.uint8) * 255
    center_x, center_y = width // 2, height // 2
    cv2.circle(vignette_center, (center_x, center_y), int(center_radius), (0, 0, 0), -1)
    return vignette_center


def cv_extend_image_horizontal(original_image, width, height):
    extend_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Extract the first and last column of pixels from the original image
    first_column = original_image[:, 0, :]
    last_column = original_image[:, -1, :]

    # Calculate the left and right padding sizes
    left_padding = (width - original_image.shape[1]) // 2
    right_padding = width - original_image.shape[1] - left_padding

    for i in range(left_padding):
        alpha = i / left_padding
        extend_image[:, i] = alpha * first_column + (1 - alpha) * np.zeros((3,), dtype=np.uint8)

    # Place the original image in the center
    extend_image[:, left_padding:left_padding + original_image.shape[1]] = original_image

    for i in range(right_padding):
        alpha = i / right_padding
        extend_image[:, -right_padding + i] = (1 - alpha) * last_column + alpha * np.zeros((3,), dtype=np.uint8)

    return extend_image


def cv_extend_image_all(original_image, width, height):
    # Create a black canvas with the new dimensions
    extended_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the offsets to center the original image
    x_offset = (width - original_image.shape[1]) // 2
    y_offset = (height - original_image.shape[0]) // 2

    # Place the original image in the center of the black canvas
    extended_image[y_offset:y_offset + original_image.shape[0],
    x_offset:x_offset + original_image.shape[1]] = original_image
    return extended_image


def cv_create_extend_canvas(width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    return canvas


def cv_place_enlarge_blur(canvas, image):
    # enlarge image to the size of canvas and kept the aspect ratio
    target_height = canvas.shape[0]
    # resize image to target height
    target_width = target_height
    # resize image to target width and height
    image = cv2.resize(image, (target_width, target_height))
    # place the iamge in the center of canvas
    x_offset = (canvas.shape[1] - image.shape[1]) // 2
    y_offset = (canvas.shape[0] - image.shape[0]) // 2
    canvas[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    # canvas[image.shape[0], x_offset:x_offset + image.shape[1]] = image
    # blur the canvas
    canvas = cv2.GaussianBlur(canvas, (51, 51), 0)
    return canvas


def place_center_image(canvas, image):
    # place image in the center of canvas
    x_offset = (canvas.shape[1] - image.shape[1]) // 2
    y_offset = (canvas.shape[0] - image.shape[0]) // 2
    canvas[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    return canvas


class StableDiffusionGenerate:
    def __init__(self,
                 controlnet_path: str,
                 diffuser_path: str,
                 inpaint_path: str,
                 lora_weights: str,
                 esrgan_weights: str,
                 prompt_generate: str,
                 prompt_inpaint: str,
                 save_hdr_prefix: str,
                 width: int = 1024,
                 height: int = 576
                 ):
        self.status = 900
        self.message = "start>"
        self.all_generated = False
        self.photo = None
        self.save_photo_prefix = None
        self.generated = None
        self.generated_path = None
        self.inpainted = None
        self.inpainted_path = None
        self.upscaled = None
        self.upscale_path = None
        self.hdri_path = None
        self.hdra_path = None
        self.controlnet_path = controlnet_path
        self.diffuser_path = diffuser_path
        self.inpaint_path = inpaint_path
        self.lora_weights = lora_weights
        self.esrgan_weights = esrgan_weights
        self.prompt_generate = prompt_generate
        self.prompt_inpaint = prompt_inpaint
        self.save_raw_prefix = "images/photo/sd_raw_{}.png"
        self.save_photo_prefix = "images/photo/sd_photo_{}.png"
        self.save_generate_prefix = "images/generate/sd_universe_{}.png"
        self.save_inpaint_prefix = "images/inpaint/sd_inpaint_{}.png"
        self.save_hdr_prefix = save_hdr_prefix + "_{}"
        self.width = width
        self.height = height
        self.planet_locations = None
        self.planet_ring_colors = None

    @classmethod
    def from_default(cls):
        CONTROLNET_PATH = "lllyasviel/sd-controlnet-canny"
        DIFFUSER_PATH = "runwayml/stable-diffusion-v1-5"
        INPAINT_PATH = "runwayml/stable-diffusion-inpainting"
        LORA_WEIGHTS = "OGX.safetensors"
        ESRGAN_WEIGHTS = "RealESRGAN_x4plus.pth"
        PROMPT_GENERATE = ("OyGalaxy A galaxy contains stars and planets, including cyan neutron stars, black holes "
                           "far away. animated, hdr, cinematic, illustration")
        PROMPT_INPAINT = "universe, hdr, nebula, vignette, black on the edge, dark, space, galaxy, stars"
        SAVE_HDR_PATH = "images/hdr/sd_hdr"
        return cls(CONTROLNET_PATH,
                   DIFFUSER_PATH,
                   INPAINT_PATH,
                   LORA_WEIGHTS,
                   ESRGAN_WEIGHTS,
                   PROMPT_GENERATE,
                   PROMPT_INPAINT,
                   SAVE_HDR_PATH)

    def take_photo(self):
        cap = cv2.VideoCapture(0)
        # Check if the camera opened successfully
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        else:
            # Capture a single frame
            ret, frame = cap.read()
            if ret:
                # Save the captured image to the path
                save_path = self.save_raw_prefix.format(int(time.time()))
                cv2.imwrite(save_path, frame)
                print(f"Image saved to {save_path}")
                cap.release()
                cv2.destroyAllWindows()
                self.photo = frame
                return save_path
            else:
                raise IOError("Cannot capture an image.")

        # Release the camera and close all windows

    def generate(self):
        path = save_image(self.sd_controlnet_i2i(self.photo, self.prompt_generate), self.save_generate_prefix)
        self.generated_path = path
        return path

    def inpaint(self):
        image, mask = self.cv_process4inpaint(self.generated_path)
        # path = save_image(self.sd_inpaint(image, mask, self.prompt_inpaint), self.save_inpaint_prefix)
        self.sd_inpaint(image, mask, self.prompt_inpaint)
        path = save_image(self.cv_mask_gradient(), self.save_inpaint_prefix)
        self.inpainted_path = path
        return path

    def upscale(self):
        path = save_image(self.gan_upscale_4x(self.inpainted_path), self.save_inpaint_prefix)
        self.upscale_path = path
        return path

    def save_hdr(self, contrast=2, brightness=150):
        image = cv2.imread(self.upscale_path, cv2.IMREAD_UNCHANGED)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_float = image.astype(np.float32)
        image_float /= 255.0
        image_float = image_float ** 2.2
        path_image = self.save_hdr_prefix.format(int(time.time())) + "_i.hdr"
        cv2.imwrite(path_image.format(int(time.time())), image_float)
        self.hdri_path = path_image
        alpha = cv2.convertScaleAbs(image, alpha=contrast)
        alpha = cv2.subtract(alpha, (brightness, brightness, brightness, 0))
        alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
        alpha = cv2.convertScaleAbs(alpha, alpha=0.5)
        # alpha = cv2.convetTo(alpha, -1, 0.5, 0)
        alpha_float = alpha.astype(np.float32)
        alpha_float /= 255.0
        path_alpha = self.save_hdr_prefix.format(int(time.time())) + "_a.hdr"
        cv2.imwrite(path_alpha.format(int(time.time())), alpha_float)
        self.hdra_path = path_alpha
        return path_image, path_alpha

    def sd_controlnet_i2i(self, photo, prompt):
        image = image_canny(photo)
        controlnet = ControlNetModel.from_pretrained(self.controlnet_path,
                                                     torch_dtype=torch.float16,
                                                     use_safetensors=True)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(self.diffuser_path,
                                                                 controlnet=controlnet,
                                                                 torch_dtype=torch.float16,
                                                                 use_safetensors=True).to("cuda")
        pipe.load_lora_weights('./models', weight_name=self.lora_weights)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        output = pipe(prompt, image=image).images[0]
        self.generated = output
        # save_image(output, self.save_generate_path)
        return output

    def sd_inpaint(self, image, mask, prompt):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(self.inpaint_path,
                                                              torch_dtype=torch.float16).to("cuda")
        output = pipe(prompt=prompt, image=image, mask_image=mask, width=self.width, height=self.height).images[0]
        # pipe.load_lora_weights('./models', weight_name=self.lora_weights)
        self.inpainted = output
        # save_image(output, self.save_inpaint_prefix)
        return output

    def cv_process4inpaint(self, generate_path):
        original_image = cv2.imread(generate_path)
        mask = cv_create_vignette_mask(self.width, self.height, center_fraction=0.7)
        canvas = cv_create_extend_canvas(self.width, self.height)
        canvas = cv_place_enlarge_blur(canvas, original_image)
        extend = place_center_image(canvas, original_image)
        init_image = cv2.cvtColor(extend.astype(np.uint8), cv2.COLOR_BGR2RGB)
        init_image = Image.fromarray(init_image)
        init_image = init_image.resize((self.width, self.height))
        mask_image = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
        mask_image = Image.fromarray(mask_image)
        mask_image = mask_image.resize((self.width, self.height))
        return init_image, mask_image

    def gan_upscale_4x(self, inpaint_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights("models/"+self.esrgan_weights, download=True)
        image = Image.open(inpaint_path).convert('RGB')
        output = model.predict(image)
        self.upscaled = output
        return output

    def cv_process_photo(self, points: list[tuple[int, int]], target_size: int = 512):
        """
        Takes a list of points that define a target area in the input photo.
        And performs a perspective transformation to align this area with a square region of the specified target size.
        The resulting transformed photo is saved and returned in RGB format.

        :param points: the points of target area
        :param target_size: the target size of transformed photo for controlnet
        """
        if self.photo is not None:
            original_points = np.float32(points)
            target_points = np.float32(
                [[0, 0], [target_size - 1, 0], [target_size - 1, target_size - 1], [0, target_size - 1]])
            matrix = cv2.getPerspectiveTransform(original_points, target_points)
            transformed = cv2.warpPerspective(self.photo, matrix, (target_size, target_size))
            save_path = self.save_photo_prefix.format(int(time.time()))
            cv2.imwrite(save_path, transformed)
            photo = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
            self.photo = photo
            return photo

    def cv_cluster_colors(self, k: int = 5):
        """
        Takes a generated image and performs k-means clustering to find the main colors of the image.

        :param k: number of target number of colors
        """
        image = self.generated
        # Reshape the image pixel values for clustering.
        image_array = np.array(image)
        pixel_values = image_array.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        # Perform k-means clustering to find main color clusters.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        # Calculate the reverse color for each cluster center.
        reverse_colors = []
        for pixel in centers:
            reverse_pixel = 255 - pixel
            reverse_colors.append(reverse_pixel)
        self.planet_ring_colors = reverse_colors
        return centers.tolist()

    def cv_calculate_centers(self, k: int, white=True):
        """
        Calculate k centers of clusters in a binary threshold image.

        :param k: number of centers
        :param white: decide whether to calculate white or black centers
        :return:
        """
        image = self.photo
        # Convert the image to grayscale and apply adaptive thresholding.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, 10, 50)
        # threshold = cv2.adaptiveThreshold(gray, 255,
        #                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # Depending on 'white' parameter, locate white or non-white regions.
        if white:
            search = np.where(canny == 255)
            search = np.column_stack((search[1], search[0])).astype(np.float32)
        else:
            search = np.where(canny != 255)
            search = np.column_stack((search[1], search[0])).astype(np.float32)
        # Perform k-means clustering to calculate cluster centers.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(search, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # centers = np.uint8(centers)
        self.planet_locations = centers
        cv2.imwrite("images/center/centers.png", canny)
        return centers.tolist()

    def cv_mask_gradient(self):
        mask = cv2.imread("myapp/test/gradient.png")
        # original = image
        inpainted = self.inpainted
        original = np.array(inpainted)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        # original = self.inpainted
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(float) / 255.0
        # mask the image
        for c in range(0, 3):
            original[:, :, c] = original[:, :, c] * mask

        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original = Image.fromarray(original)
        self.inpainted = original
        return original

    def set_status(self, status, description):
        self.status = status
        self.message = description

