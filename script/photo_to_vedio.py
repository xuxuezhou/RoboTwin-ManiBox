import imageio
import os

def create_video(image_folder, output_video_file, save_dir='./task_video/', fps=30):
    video_dir = save_dir
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('.')[0]))

    writer = imageio.get_writer(os.path.join(video_dir, output_video_file), fps=fps)

    for i, image_name in enumerate(images, start=1):
        img_path = os.path.join(image_folder, image_name)
        try:
            image = imageio.imread(img_path)
            writer.append_data(image)
            print(f"Added {image_name} to video ({i}/{len(images)})", end='\r')
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    writer.close()
    print("\nVideo creation complete.")

if __name__ == '__main__':
    task_list = ['mug_hanging']
    load_dir = './data'
    for task in task_list:
        create_video(f'{load_dir}/{task}/episode0/camera/color/right/',f'{task}_right.mp4', save_dir='./task_video/')
