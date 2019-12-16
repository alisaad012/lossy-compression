#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
import re
from tqdm import tqdm

import model, sample, encoder

import synonyms
synonyms.setup()

def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = yield
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    yield text

def parsing(input_file, output_file, encode=True, overwrite=False):
    prediction_list = interact_model(
        model_name='124M',
        seed=0,
        nsamples=1,
        batch_size=1,
        length=1,
        temperature=1,
        top_k=0,
        top_p=1,
        models_dir='models',
    )

    verb = "Encoding" if encode else "Decoding"
    print(f"{verb} {input_file} to {output_file}")

    if overwrite:
        with open(output_file, 'w+') as f_out:
            f_out.write('')
    else:
        import os
        if output_file in os.listdir('.'):
            return

    output = ""
    with open(input_file, 'r') as f_in:
        source = f_in.read()
        flag = "\x1a"
        if encode:
            assert flag not in source
        else:
            if flag not in source:
                print("No flags")
                with open(output_file, 'w+') as f_out:
                    f_out.write(source)
                    return
        parsed_source = re.split(r"([\s.]*\.[\s.]*)", source.strip() + '.\n')

        with tqdm(ncols=100, desc="Sentence Progress ", total=len(parsed_source) // 2, unit_scale=True) as pbar:
            for i, (sentence, separator) in enumerate(zip(parsed_source[::2], parsed_source[1::2])):
                tokens = sentence.split()
                prephrased = ''
                postphrased = ''

                if len(tokens) == 0:
                    pass
                elif len(tokens) == 1:
                    prephrased = tokens[0]
                    postphrased = tokens[0]
                else:
                    prephrased = tokens[0]
                    postphrased = tokens[0]

                    for index in range(1, len(tokens)):
                        token = tokens[index]

                        if encode:
                            next(prediction_list)
                            prediction = prediction_list.send(prephrased).strip()
                            prephrased += ' ' + token
                            if prediction == token or synonyms.is_synonym(prediction, token):
                                postphrased += ' ' + flag
                            else:
                                postphrased += ' ' + token
                        else:
                            next(prediction_list)
                            prediction = prediction_list.send(prephrased).strip()
                            if token == flag:
                                prephrased += ' ' + token
                                postphrased += ' ' + prediction
                            else:
                                prephrased += ' ' + token
                                postphrased += ' ' + token
                
                with open(output_file, 'a+') as f_out:
                    if i == len(parsed_source) // 2 - 1:
                        separator = separator[:-2] + '\n'
                    f_out.write(postphrased + separator)
            pbar.update(1)

if __name__ == '__main__':
    fire.Fire(parsing)
