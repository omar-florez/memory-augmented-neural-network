import numpy as np
import PIL.Image as Image
import ipdb

def load_image_as_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def extract_regions(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=0.3):

    regions = {}
    image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
    width, height = image_pil.size

    for i in range(boxes.shape[0]):
        if scores is None or scores[i]>min_score_thresh:
            box = boxes[i]
            ymin, xmin, ymax, xmax = boxw

            if use_normalized_coordinates:
                left, right, top, bottom = (xmin * width, xmax * width,
                                            ymin * height, ymax * height)
            else:
                left, right, top, bottom = (xmin, xmax, ymin, ymax)

            region = {}
            region['image'] = image_pil.crop((int(left), int(top), int(right), int(bottom)))
            if scores is not None:
                region['score'] = scores[i]

            class_name = category_index[classes[i]]['name']
            if class_name not in regions:
                regions[class_name] = []
            regions[class_name].append(region)
    return regions

