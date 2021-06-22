import os
import argparse


parser = argparse.ArgumentParser(description='Prepare data')
parser.add_argument('-f', '--xslx_pth', type=str,
                    default='data/recorded_data.xlsx',
                    help="Path excel with the recorded data")




def main(args):
    pass


if __name__ == '__main__':
    main(parser.parse_args())