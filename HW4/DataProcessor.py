from PIL import Image
import os
import numpy as np
from string import digits
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, img_as_float

class DataProcessor:
    def __init__(self, current_directory):
        self.current_directory = current_directory
        self.data = None
        self.size = (64, 64)
        self.mean_image = None

    def load_images(self):
        data = []
        jaffe_path = os.path.join(self.current_directory, 'jaffe')
        output_path = os.path.join(self.current_directory, 'resized_jaffe')
        # os.makedirs(output_path, exist_ok=True)

        for image_file in os.listdir(jaffe_path):
            try:
                image_path = os.path.join(jaffe_path, image_file)
                img = io.imread(image_path)
                img = img_as_float(img)
                resized_img = Image.fromarray(img).resize(self.size)
                # output_filepath = os.path.join(output_path, image_file)
                # resized_img.save(output_filepath)

                img_array = np.array(resized_img).flatten()
                remove_digits = str.maketrans('', '', digits)
                expression = image_file.split('.')[1].translate(remove_digits)
                subject = image_file.split('.')[0]

                data.append({
                    'Subject': subject,
                    'Expression': expression,
                    'Image': img_array
                })
            except (IOError, OSError) as e:
                continue

        self.data = pd.DataFrame(data)

    def plot_image(self, img, size):
        image_array = img.reshape(size)
        plt.figure(figsize=(4, 4))
        plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        plt.show()

    def calculate_mean_image(self):
        if self.data is not None:
            images = self.data['Image'].tolist()
            sum_images = np.zeros(len(images[0]))
            for img in images:
                sum_images += (img)
            self.mean_image = sum_images / len(images)
            self.plot_image(self.mean_image, self.size)
        else:
            print("Error: Images not loaded.")

    def demean_images(self):
        if self.mean_image is not None and self.data is not None:
            demeaned_images = []
            for img in self.data['Image']:
                demeaned_img = img - self.mean_image
                demeaned_images.append(demeaned_img)
            self.data['Normalized_Image'] = demeaned_images
        else:
            print("Error: Mean image or images not available.")
            
    def normalize_images(self):
        if self.mean_image is not None and self.data is not None:
                normalized_images = []
                for img in self.data['Normalized_Image']:
                    normalized_img = img / 255.0
                    normalized_images.append(normalized_img)
                self.data['Normalized_Image'] = normalized_images
        else:
            print("Error: Mean image or images not available.")
        
    def visualize_selected_images(self):
        if self.data is not None:
            unique_classes = self.data['Expression'].unique()
            selected_images = []

            for emotion in unique_classes:
                emotion_data = self.data[self.data['Expression'] == emotion]
                selected_row = emotion_data.sample(1).iloc[0]
                selected_images.append(selected_row)

            fig, axes = plt.subplots(nrows=2, ncols=len(selected_images), figsize=(len(selected_images) * 4, 8))

            for i, row in enumerate(selected_images):
                original_image = row['Image'].reshape(self.size)
                demeaned_image = row['Normalized_Image'].reshape(self.size)

                axes[0, i].imshow(original_image, cmap='gray')
                axes[0, i].set_title(f"Original - {row['Expression']}")
                axes[0, i].axis('off')

                axes[1, i].imshow(demeaned_image, cmap='gray')
                axes[1, i].set_title(f"Demeaned - {row['Expression']}")
                axes[1, i].axis('off')

            plt.show()
        else:
            print("Error: Images not loaded.")

