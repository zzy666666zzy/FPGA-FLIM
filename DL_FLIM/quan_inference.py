import torch
from Q_adder_basic import Quan_adder2d
from Q_S_TauNet_AVE_model import Quan_S_TauNet_AVE


PATH=r"./Q_addernet_pth/good_litemodel_loss_0.046051.pth"
model = Quan_S_TauNet_AVE()
checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint,strict=False)

model.eval()
for m in model.modules():
    if isinstance(m, Quan_adder2d):
        m.Weight.data = m.weight_quantizer(m.Weight)