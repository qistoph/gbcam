# gbcam

"Stream like a GameBoy Camera."

gbcam provides a virtual camera that filters a real camera stream to make it look like a GameBoy Camera.

![gbcam Preview](/images/gbcam_preview1.jpg)

## Installation

```
# Clone the rope
git clone https://www.github.com/qistoph/gbcam

# Enter project
cd gbcam

# Install requirements
pip3 install -r requirements.txt

# Create fake video device
sudo modprobe v4l2loopback devices=1
```

## Usage

Make sure the `camera` and `fakecam` in `config.yaml` are pointing to the right devices.

```
python3 filter.py
```

## Inspiration

- https://petapixel.com/2020/09/04/this-guy-turned-his-game-boy-camera-into-a-functional-webcam/
- https://maple.pet/webgbcam/

## Using

- [FreeSimpleGUI](https://github.com/spyoungtech/FreeSimpleGUI)
- [pyfakewebcam](https://pypi.org/project/pyfakewebcam/)
- [OpenCV](https://opencv.org/)
- [oyaml](https://pypi.org/project/oyaml/)
