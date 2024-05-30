import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # out path
    parser.add_argument('--input', type=str, default='', help='input path')
    parser.add_argument('--output', type=str, default='', help='output path')
    parser.add_argument('--type', type=str, default='bio', help='type of processing required bio/tsv')

    args = parser.parse_args()
    return args

args = get_arguments()


if args.type == 'bio':
    print(args.input)
    with open(args.input, 'r') as f:
        a = f.readlines()

    with open(args.output, 'w') as f:
        sent = []
        label = []
        for i in a:
            if i.strip() == '':
                if len(sent):
                    f.write(' '.join(sent)+'\n')
                    sent = []
                    label = []
                continue

            sent.append(i.strip().split('\t')[0])
            label.append(i.strip().split('\t')[1])

if args.type=='tsv':
    with open(args.input, 'r') as f:
        a = f.readlines()

    with open(args.output, 'w') as f:
        for i in a[1:]:
            if len(i.strip().split('\t'))==1:
                f.write('\n')
            else:
                f.write(i.strip().split('\t')[0]+'\n')

if args.type=='sim':
    with open(args.input, 'r') as f:
        a = f.readlines()

    with open(args.output, 'w') as f:
        for i in a[1:]:
            if len(i.strip().split('\t'))==1:
                f.write('\n')
            else:
                f.write(i.strip().split('\t')[0]+'\n')