import numpy
import hvwfg

def cal_ps_hv(pf, pf_num, ref):
    batch_size = pf.size(0)
    device = pf.device
    # hvs = []
    hvs = numpy.zeros([batch_size, 1])
    ref_region = 1
    for i in range(ref.shape[0]):
        ref_region = ref_region * ref[i]
    for k in range(batch_size):
        num = pf_num[k]
        # cur_pf_sort = pf[k].sum(-1).argsort(-1)
        # cur_pf = pf[k][cur_pf_sort[:num+1]]
        hv = hvwfg.wfg(pf[k][:num].cpu().numpy().astype(float), ref.astype(float))
        # hv = hv / (ref[0] * ref[1])
        hv = hv / ref_region
        # hvs.append(hv)
        hvs[k] = hv

    # hvs = torch.FloatTensor(hvs).to(device=device)
    # hvs = hvs.reshape(batch_size, -1)
    return hvs
