import argparse
from learning_racer.commands.subcommand import command_demo, command_train
from learning_racer.config import ConfigReader

from logging import getLogger

logger = getLogger(__name__)

__version__ = '1.6.0'

parser = argparse.ArgumentParser(description='Learning Racer command.')
parser.add_argument('--version', action='version', version='learning_racer version {} .'.format(__version__))
subparser = parser.add_subparsers()

# train subcommand.
parser_train = subparser.add_parser('train', help='see `train -h`')
parser_train.add_argument('-config', '--config-path', help='Path to a config.yml path.',
                          default='config.yml', type=str)
parser_train.add_argument('-vae', '--vae-path', help='Path to a trained vae model path.',
                          default='vae.torch', type=str)
parser_train.add_argument('-device', '--device', help='torch device {"cpu" | "cuda"}',
                          default='cuda', type=str)
parser_train.add_argument('-robot', '--robot-driver', help='choose robot driver from {"jetbot", "jetracer", "sim"}',
                          default='jetbot', type=str)
parser_train.add_argument('-steps', '--time-steps', help='total step.',
                          default='5000', type=int)
parser_train.add_argument('-save_freq', '--save-freq-steps', help='number of step for each checkpoints.',
                          default='1000', type=int)
parser_train.add_argument('-save_path', '--save-model-path', help='Model save path.',
                          default='model_log', type=str)
parser_train.add_argument('-s', '--save', help='save model file name.',
                          default='model', type=str)
parser_train.add_argument('-l', '--load-model', help='Define pre-train model path.',
                          default='', type=str)
parser_train.add_argument('-log', '--tb-log', help='Define logging directory name, If not set, Do not logging.',
                          default=None, type=str)
parser_train.set_defaults(handler=command_train)

# demo subcommand.
parser_demo = subparser.add_parser('demo', help='see `demo -h`')
parser_demo.add_argument('-config', '--config-path', help='Path to a config.yml path.',
                         default='config.yml', type=str)
parser_demo.add_argument('-vae', '--vae-path', help='Path to a trained vae model path.',
                         default='vae.torch', type=str)
parser_demo.add_argument('-model', '--model-path', help='Path to a trained vae model path.',
                         default='model', type=str)
parser_demo.add_argument('-device', '--device', help='torch device {"cpu" | "cuda"}',
                         default='cuda', type=str)
parser_demo.add_argument('-robot', '--robot-driver', help='choose robot driver',
                         default='jetbot', type=str)
parser_demo.add_argument('-steps', '--time-steps', help='total step.',
                         default='5000', type=int)
parser_demo.add_argument('-user', '--sim-user', help='Define user name for own car that showed DonkeySim',
                         default='anonymous', type=str)
parser_demo.add_argument('-car', '--sim-car', help='Define car model type for own car that showed DonkeySim',
                         default='Donkey', type=str)
parser_demo.add_argument('-log', '--tb-log', help='Define logging directory name, If not set, Do not logging.',
                         default=None, type=str)
parser_demo.set_defaults(handler=command_demo)


def racer_func():
    config = ConfigReader()
    args = parser.parse_args()
    config.load(args.config_path)

    if hasattr(args, 'handler'):
        try:
            args.handler(args, config)
        except Exception as ex:
            raise ex
    else:
        parser.print_help()


if __name__ == '__main__':
    racer_func()
