# Transfer Learning Examples

### train_vgg16 fine tune

fine modified the output of vgg16 use trained parameters, modified the output from 1000 to 2
Also, modified the linear layer 4096 to save memory

*python3 no longer support / for integer to tensor use torch.true_divide or float or // instead*
 