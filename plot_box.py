import numpy as np
from PIL import Image, ImageFont, ImageDraw


def show_box(img, boxes, scores):
    image = Image.fromarray(img).copy()
    print(image.size)
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype(
                                  'int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in enumerate(scores):
        # predicted_class = self.class_names[c]
        box = boxes[i]
        score = scores[i]

        label = '{:.2f}'.format(score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([top - label_size[1], left])
        else:
            text_origin = np.array([top + 1, left])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [top + i, left + i, bottom - i, right - i])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
        # break
    image.show()
