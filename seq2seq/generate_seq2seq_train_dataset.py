import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='semeval15')
args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv('data/{}_train.csv'.format(args.dataset))
    sample_ids = set(df.sample_id.tolist())
    output_template = '("{aspect_category}", "{sentiment}")'
    inputs_list = []
    outputs_list = []
    for sampleid in sample_ids:
        subdf = df[df['sample_id'] == sampleid]
        curr_text = list(set(subdf.text.tolist()))
        inputs_list.append(curr_text[0])
        curr_output = None
        for idx, row in subdf.iterrows():
            if curr_output is None:
                curr_output = output_template.format(aspect_category=str(row['aspect_category']), sentiment=str(row['sentiment']))
            else:
                curr_output = curr_output + '; ' + output_template.format(aspect_category=str(row['aspect_category']), sentiment=str(row['sentiment']))
        outputs_list.append(curr_output)
    outdf = pd.DataFrame({'input_string': inputs_list, 'output_string': outputs_list})
    outdf.to_csv('data/trainset_seq2seq_{}.csv'.format(args.dataset))
