import json
import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

DEBUG = False
def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions, img_ids) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field, label):
    import itertools
    model.eval()
    gen, gen_w_ids = {}, {}
    gts, gts_w_ids = {}, {}
    image_ids = set()
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt, img_ids) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen_i_wo_whitespace = ''.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
                # if img_ids[i] not in image_ids:
                gen_w_ids[img_ids[i]] = gen_i_wo_whitespace.replace(" ","")
                gts_w_ids[img_ids[i]] = list(map(lambda x: x.replace(" ",""), gts_i))
                image_ids.add(img_ids[i])
            pbar.update()
            if DEBUG and it > 10:
                break

    # gts = evaluation.PTBTokenizer.tokenize(gts)
    # gen = evaluation.PTBTokenizer.tokenize(gen)

    skey = list(gts.keys())[1]
    print("hypo:", gen[skey],flush=True)
    print("refs:", gts[skey],flush=True)
    with open(f"output_{label}.json","w") as f:
        results = [{"image_id" : img_id, "caption" : cap, "gt": gts_w_ids[img_id]} for img_id, cap in gen_w_ids.items()]
        output = json.dumps(results,indent=4)
        f.write(output)

    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions, img_ids) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            out = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()
            if DEBUG and it > 5:
                break

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt, img_ids) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline

def set_UTF8_io():
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

if __name__ == '__main__':
    set_UTF8_io()
    print("日本語 ... OK",flush=True)

    assert torch.cuda.is_available()
    print("torch:",torch.__version__,flush=True)
    device = torch.device('cuda')
    # device = torch.device('cpu')
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    args = parser.parse_args()
    print(args,flush=True)

    print('Meshed-Memory Transformer Training',flush=True)


    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary",flush=True)
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
        print("..fin",flush=True)
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    # Model and dataloaders
    print("Prepare Model ...",flush=True)
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': args.m})
    decoder = MeshedDecoder(len(text_field.vocab), 100 - 1, 3, text_field.vocab.stoi['<pad>']) # check https://yuiga.dev/blog/posts/pytorchcuda_error_device-side_assert_triggered_%E8%A7%A3%E6%B1%BA%E3%81%AE%E6%89%8B%E5%BC%95%E3%81%8D/
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder)
    model = model.to(device)

    print("Prepare dataset ... ",flush=True)
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    print("Prepare token ... ",flush=True)
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)


    # Initial conditions
    print("Prepare initial conditions ...",flush=True)
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            def __safe_inject(key,lam):
                if key not in data:
                    return
                lam(data[key])
                return data[key]

            def __safe_return(key,x):
                return data[key] if key in data else x

            __safe_inject("torch_rng_state",torch.set_rng_state)
            __safe_inject("cuda_rng_state",torch.cuda.set_rng_state)
            __safe_inject("numpy_rng_state",np.random.set_state)
            __safe_inject("random_rng_state",random.setstate)
            __safe_inject("optimizer",optim.load_state_dict)
            __safe_inject("scheduler",scheduler.load_state_dict)

            start_epoch = __safe_return("epoch",start_epoch)
            best_cider = __safe_return("best_cider",best_cider)
            patience = __safe_return("patience",patience)
            use_rl = __safe_return("use_rl",use_rl)

            model.load_state_dict(data['state_dict'], strict=True)
            print('Resuming from epoch %d,  and best cider %f' % (start_epoch, best_cider))

    if DEBUG:
        use_rl = False

    print("Training starts",flush=True)
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

        if not use_rl:  
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            # writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, text_field)
            # writer.add_scalar('data/train_loss', train_loss, e)
            # writer.add_scalar('data/reward', reward, e)
            # writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = 1e10
        if not DEBUG:
            val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        # writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field, label="eval")
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        # writer.add_scalar('data/val_cider', val_cider, e)
        # writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        # writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        # writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        # writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field, label="test")
        print("Test scores", scores)
        # writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        # writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        # writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        # writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        # writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load('saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s_last.pth' % args.exp_name)

        if best:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)

        if exit_train:
            # writer.close()
            break
