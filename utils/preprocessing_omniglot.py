from PIL import Image
import glob

# download the datasets into the the omniglot folder and do the decrompress them
# https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip
# https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip
if __name__ == '__main__':
    dataset_folder = '../datasets/omniglot/'
    all_images = glob.glob(dataset_folder + '*/*/*/*', recursive=True)

    for i, image_filename in enumerate(all_images):
        im = Image.open(image_filename)
        im = im.resize((28, 28), resample=Image.LANCZOS)
        im.save(image_filename)
        if i % 100 == 0:
            print(i, '...', end=' ')
