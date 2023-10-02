import os

for image in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if image.endswith(".jpg"):
        # Change extension to png
        image_name = image.split(".")[0]
        os.rename(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), image),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                image_name + ".png",
            ),
        )
