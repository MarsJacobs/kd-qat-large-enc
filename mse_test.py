import torch
from torch.nn import MSELoss

kl_student_atts = torch.load("cola_kl_st.pth")
two_student_atts = torch.load("cola_two_st.pth")
teacher_atts = torch.load("cola_kl_tc.pth")

tmp_mse = 0
kl_loss = 0
two_loss = 0

loss_mse = MSELoss()  


for i, (student_att, teacher_att) in enumerate(zip(kl_student_atts, teacher_atts)):
    
    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(student_att),
                                        student_att)
    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(teacher_att),
                                teacher_att)
    import pdb; pdb.set_trace()                 
    tmp_loss = loss_mse(student_att, teacher_att)
    kl_loss += tmp_loss
print(f"kl loss is {kl_loss}")

two_loss = 0
for i, (student_att, teacher_att) in enumerate(zip(two_student_atts, teacher_atts)):
    
    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(student_att),
                                        student_att)
    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(teacher_att),
                                teacher_att)
                                
    tmp_loss = loss_mse(student_att, teacher_att)
    two_loss += tmp_loss

print(f"two loss is {two_loss}")
