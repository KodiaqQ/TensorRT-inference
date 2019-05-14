import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import time
from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
import torch.onnx as onnx
import torch

if __name__ == '__main__':
    cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_1x.py')
    cfg.model.pretrained = None

    # construct the model and load checkpoint
    model = build_detector(cfg.model)
    _ = load_checkpoint(model,
                        'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')

    model.eval()
    model = model.cuda()
    # test a single image
    # img = mmcv.imread('../data/1.jpg')
    # start = time.time()
    # result = inference_detector(model, img, cfg)
    # stop = time.time()
    # print('inf time - pyt %s' % (stop - start))
    # show_result(img, result)

    img_transform = ImageTransform(size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    dummy_input = torch.randn(10, 3, 224, 224).int()

    img_list = []
    img_meta = []

    for i, tmp_im in enumerate(dummy_input):
        ori_shape = tmp_im.shape
        tmp_im = tmp_im.cuda()
        tmp_img, img_shape, pad_shape, scale_factor = img_transform(
            tmp_im,
            scale=cfg.data.test.img_scale,
            keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
        img_list.append(to_tensor(tmp_img).cuda().unsqueeze(0))
        img_meta.append(
            dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=False)
        )

    test_input = dict(img=img_list, img_meta=img_meta)

    input_names = ["img", "img_meta"]
    output_names = ["output1"]

    torch.onnx.export(model, test_input, "f-rcnn.onnx", verbose=True, input_names=input_names,
                      output_names=output_names)
