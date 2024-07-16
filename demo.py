import json
import cv2
import gradio as gr
from PIL import Image
#
from ultralytics import YOLO
#
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

#
model = YOLO('./model/best.pt')
#
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cpu'
detector = Predictor(config) # Dùng để nhận dạng hóa đơn


def OCR_bill(img_input):
  result = model.predict(img_input)
  list_names = result[0].names
  list_bbox = result[0].boxes.data
  dict_invoice = {}
  for bbox in list_bbox:
    x_min, y_min, x_max, y_max, conf, cls = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    content_img = img_input[y_min:y_max, x_min:x_max]
    temp1 = Image.fromarray(content_img)
    content = detector.predict(temp1)
    name = list_names[int(cls)]
    if name not in dict_invoice.keys():
      dict_invoice[name] = [content]
    else:
      dict_invoice[name].append(content)
  return json.dumps(dict_invoice, indent=1)


def gradio_OCR(img):
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    result = OCR_bill(img_cv2)
    return result

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_OCR,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="OCR Result", type="json", default="{}"),
    title="Invoice OCR",
    description="Upload an image of an invoice and extract the text."
)

# Launch Gradio app
if __name__ == "__main__":
    iface.launch()




