# import torch

# BATCH_SIZE = 16
# MAX_SEQ = 32
# NUM_HEADS = 4

# teacher = torch.rand((BATCH_SIZE, NUM_HEADS, MAX_SEQ, MAX_SEQ))  # dummy attn prob after softmax.
# student = torch.rand((BATCH_SIZE, NUM_HEADS, MAX_SEQ, MAX_SEQ))  # dummy attn prob after softmax.
# mask = torch.bernoulli(torch.ones(BATCH_SIZE, NUM_HEADS, MAX_SEQ, MAX_SEQ), p=0.5).bool()  # dummy valid attn mask
# # mask 1 = valid, 0 = invalid
# length = torch.randint(1, MAX_SEQ, (BATCH_SIZE,)).long()  # valid length
# import pdb; pdb.set_trace()
# # KLD(teacher || student)
# # = sum (p(t) log p(s)) - sum(p(t) log p(t))

# teacher = torch.clamp_min(teacher, 1e-8)  # prevent 0 in log
# student = torch.clamp_min(student, 1e-8)  # prevent 0 in log

# # p(t) log p(s)
# cross_entropy = teacher * torch.log(student) * mask
# cross_entropy = torch.sum(cross_entropy, dim=-1)  # (b, h, s, s) -> (b, h, s)
# cross_entropy = torch.sum(cross_entropy, dim=-1) / length.view(-1, 1)  # (b, h, s) -> (b, h)

# # p(t) log p(t)
# entropy = teacher * torch.log(teacher) * mask
# entropy = torch.sum(entropy, dim=-1)  # (b, h, s, s) -> (b, h, s)
# entropy = torch.sum(entropy, dim=-1) / length.view(-1, 1)  # (b, h, s) -> (b, h)

# kld_loss = entropy - cross_entropy  # (b, h)
# kld_loss = torch.mean(kld_loss)  # average over heads and batch

import torch

BATCH_SIZE = 32
MAX_SEQ = 64
NUM_HEADS = 4

teacher_prob = torch.rand((BATCH_SIZE, NUM_HEADS, MAX_SEQ, MAX_SEQ), dtype=torch.float32)  # dummy prob.
student_prob = torch.rand((BATCH_SIZE, NUM_HEADS, MAX_SEQ, MAX_SEQ), dtype=torch.float32)  # dummy prob.
length = torch.randint(MAX_SEQ // 2, MAX_SEQ, (BATCH_SIZE,)).long()  # valid length
mask = torch.zeros(BATCH_SIZE, NUM_HEADS, MAX_SEQ, MAX_SEQ, dtype=torch.float32)
# mask 1 = valid, 0 = invalid

for i in range(BATCH_SIZE):
    s = length[i]
    teacher_prob[i, :, s:, :] = 0.0
    teacher_prob[i, :, :, s:] = 0.0
    student_prob[i, :, s:, :] = 0.0
    student_prob[i, :, :, s:] = 0.0

    teacher_prob[i, :, :s, :s] = torch.softmax(teacher_prob[i, :, :s, :s], dim=-1)
    student_prob[i, :, :s, :s] = torch.softmax(student_prob[i, :, :s, :s], dim=-1)
    mask[i, :, :s, :s] = 1.0

# ------------------------------------------------------------------------------------  #
# KLD(teacher || student)
# = sum (p(t) log p(t)) - sum(p(t) log p(s))
# = (-entropy) - (-cross_entropy)

teacher = torch.clamp_min(teacher_prob, 1e-8)  # prevent 0 in log
student = torch.clamp_min(student_prob, 1e-8)  # prevent 0 in log

# p(t) log p(s) = negative cross entropy
neg_cross_entropy = teacher * torch.log(student) * mask
neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1)  # (b, h, s, s) -> (b, h, s)
neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1) / length.view(-1, 1)  # (b, h, s) -> (b, h)

# p(t) log p(t) = negative entropy
neg_entropy = teacher * torch.log(teacher) * mask
neg_entropy = torch.sum(neg_entropy, dim=-1)  # (b, h, s, s) -> (b, h, s)
neg_entropy = torch.sum(neg_entropy, dim=-1) / length.view(-1, 1)  # (b, h, s) -> (b, h)

kld_loss = neg_entropy - neg_cross_entropy  # (b, h)
kld_loss = torch.mean(kld_loss)  # average over heads and batch

# KLD loss should be in range [0, inf]
print(kld_loss.item())

# ------------------------------------------------------------------------------------  #
# compare to torch KLD loss
kld_loss_orig = []
for i in range(BATCH_SIZE):
    for j in range(NUM_HEADS):
        s = length[i]
        log_teacher = torch.log(teacher_prob[i, j, :s, :s])
        log_student = torch.log(student_prob[i, j, :s, :s])

        # both are same
        kld = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")(log_student, log_teacher)
        # kld = torch.nn.KLDivLoss(log_target=False, reduction="batchmean")(log_teacher, log_student.exp())
        kld_loss_orig.append(kld)

kld_loss_orig = torch.stack(kld_loss_orig)
kld_loss_orig = kld_loss_orig.mean()
print(kld_loss_orig.item())

#Message Kyuhong Shim






