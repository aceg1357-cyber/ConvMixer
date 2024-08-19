from abu_run_copy import run
in_channel = 21
pred_len_list = [96,192,336,720]
out_channel_list = [32]
kernal_size2pad = {16:(0,15)}
dropout_rate = [0.1]
for pred_len in pred_len_list:
    for out_channel in out_channel_list:
        for k,v in kernal_size2pad.items():
            run(pred_len=pred_len,in_channel = in_channel,out_channel=out_channel,kernel_size=k,pad=v)
print("end!")
