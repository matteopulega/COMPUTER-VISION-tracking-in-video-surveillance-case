import warnings
import argparse

from main import main_sort, main_mct


def program():
    parser = argparse.ArgumentParser()
    parser.add_argument('modality', choices=['sort', 'mct'],
                        help='This script works in two modality of tracking: sort and mct. Select one of those.')
    parser.add_argument('input_video', help='Path to the input video.')
    parser.add_argument('output_video',
                        help='Path to the output video. If it contains multiple folders, be sure that all the folders'
                             'already exist. Do not pass the extension, we only create .avi videos.')
    args = parser.parse_args()
    if args.modality == 'sort':
        main_sort(args.input_video, args.output_video)
    else:
        main_mct(args.input_video, args.output_video)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        program()
