from tqdm import tqdm
import json
import cv2
import os

dump_obj = {}

dist = ""

grid_sizes = [2, 4, 5, 8, 10, 16]


# walks a directory and gets similarity scores for each image therein
# expects files to be images, will probably complain if they're not
def sim_index_scan(target_dir):
    global dump_obj
    dump_obj = {}

    for file in tqdm(os.listdir(target_dir)):
        if os.path.isdir(os.path.join(target_dir, file)):
            continue
        dump_obj[file] = {}
        for size in grid_sizes:

            num_scores = int(size / 2)

            similarity_scores = []
            similarity_dict = {}

            # Load the original image
            img = cv2.imread(os.path.join(target_dir, file))

            # Split the image into sub-images
            sub_images = []
            for i in range(size):
                for j in range(size):
                    sub_img = img[i * img.shape[0] // size:(i + 1) * img.shape[0] // size,
                                  j * img.shape[1] // size:(j + 1) * img.shape[1] // size]
                    sub_images.append(sub_img)

            # Compare each sub-image to every other sub-image except for the sub-images' orthogonal neighbors
            for i in range(len(sub_images)):
                for j in range(i+1, len(sub_images)):
                    if i == j:
                        continue
                    if size > 2 and i // size == j // size or i % size == j % size:
                        continue
                    hist1 = cv2.calcHist([sub_images[i]], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([sub_images[j]], [0], None, [256], [0, 256])
                    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
                    # print(f"Similarity between sub-image {i} and sub-image {j}: {similarity}")
                    similarity_scores.append(similarity)
                    similarity_dict[f"{i},{j}"] = similarity

            similarity_scores.sort()

            length = len(similarity_scores)
            half_length = length/2

            dump_obj[file][size] = {"avg_distance": sum(similarity_scores) / length,
                                    "total_distance": similarity_scores[-1] - similarity_scores[0],
                                    "avg_distance_high": sum(similarity_scores[num_scores:]) / half_length,
                                    "avg_distance_low": sum(similarity_scores[0:num_scores]) / half_length,
                                    "similarity_scores": similarity_scores,
                                    "similarity_dict": dict(sorted(similarity_dict.items(), key=lambda x: x[1]))}

    return dump_obj


json_obj = {
    'correct': sim_index_scan("correct_cropped"),
    'incorrect': sim_index_scan("incorrect_cropped")
}


with open('data.json', 'w') as file:
    json.dump(json_obj, file, indent=4)
