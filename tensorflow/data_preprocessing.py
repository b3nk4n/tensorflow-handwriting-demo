import sys
import argparse
import json
import urllib.request


def main(_):
    """Executed only if run as a script."""
    url = 'http://localhost:3000/api/handwriting'
    response = urllib.request.urlopen(url)
    handwriting_list = json.loads(response.read().decode())
    print(handwriting_list)
    print(len(handwriting_list))

    for handwriting in handwriting_list:
        print(handwriting['label'])

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--batch_size', type=int, default=32,
                        help='The batch size')
    FLAGS, UNPARSED = PARSER.parse_known_args()
    main(FLAGS)
    #tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
