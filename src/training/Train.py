from Models import provaModel as pM


def train_or_eval_model(model, loss_fun, dataloader, epoch, optimizer=None, train=False ):
    #assert not train or optimizer != None
    """
    if train:
        model.train()
    else:
        model.eval()
    """

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        text_feature, audio_feature, qmask, umask, labels, vid = data


        text_feature = text_feature.permute(0, 1, 3, 2).squeeze(-1)
        audio_feature = audio_feature.permute(0, 1, 3, 2).squeeze(-1)
        labels = labels.squeeze(-1)
        umask = umask.permute(1, 0)
        qmask = qmask.permute(0, 1, 3, 2)[:, :, 0, 0]

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))] # Compute the real length of a sequence

        t_log_prob, a_log_prob, all_log_prob, all_prob = model(text_feature, audio_feature, qmask, umask)
        print(f"t_log_prob: {t_log_prob}\n a_log_prob: {a_log_prob} \n all_log_prob: {all_log_prob} \n all_prob: {all_prob}")