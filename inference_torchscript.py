import cv2, torch, time
import numpy as np
from torchvision import transforms
from PIL import Image
from torch import jit


device = torch.device('cuda:0')
model = jit.load("model.pt")
model.to(device).eval()

cap = cv2.VideoCapture("d:\\video\\IMG_1580.MOV")
fps = 0.0
t0 = time.time()

I_H = 540
I_W = 960


@torch.no_grad()
def main():
    global fps, t0

    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()

        if ret:
            h, w = frame.shape[0:2]
            image_data = Image.fromarray(np.uint8(frame))

            img_tensor = transforms.functional.to_tensor(
                transforms.functional.resized_crop(image_data, h - w // 2, 0, w // 2, w, (I_H, I_W)))
            img_tensor.unsqueeze_(0)
            output = model.forward(img_tensor.to(device))

            out = torch.sigmoid(output['out'])
            final_out = torch.argmax(out[0], 0)
            final_out = final_out.cpu().numpy().astype(np.uint8)
            final_out *= 255
            final_out = cv2.cvtColor(final_out, cv2.COLOR_GRAY2BGR)

            frame_2 = cv2.resize(frame, (I_W, I_H), interpolation=cv2.INTER_AREA)

            fps = 1 / (time.time() - t0)
            print("fps-image_processing: ", fps)

            result_frame = cv2.addWeighted(src1=final_out, alpha=0.5, src2=frame_2, beta=1., gamma=0.)
            cv2.imshow("frame", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()