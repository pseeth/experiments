import separate_music
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_directory")
    parser.add_argument("--folder")
    args = parser.parse_args()

    tracks = sorted([os.path.join(args.folder, x) for x in os.listdir(args.folder)])

    for track in tracks:
        print(track)
        try:
            separate_music.run(args.run_directory, track, 'cuda')
        except:
            separate_music.run(args.run_directory, track, 'cpu')