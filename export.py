import torch, json
from model import SCNN


device = torch.device('cuda:0')
weight_path = './weights/mobilenet_v2_test_best.pth'

net = SCNN(input_size=(512, 288), pretrained=False)
net.to(device)
net.eval()
save_dict = torch.load(weight_path, map_location=device)
weights = save_dict['net']
net.load_state_dict(weights)


def export_torchscript(model, im, file):
    ts = torch.jit.trace(model, im, strict=False)
    #d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
    #extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
    ts.save(str(file))


image = torch.zeros(1, 3, 512, 288).to(device)  # image size(Batch, Color, Height, Width)
export_torchscript(model=net, im=image, file='./weights/mobilenet_v2_test_best.torchscript')

