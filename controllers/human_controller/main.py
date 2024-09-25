import optparse
import math



def func():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--trajectory", default="", help="Specify the trajectory in the format [x1 y1, x2 y2, ...]")
    opt_parser.add_option("--speed", type=float, default=0.5, help="Specify walking speed in [m/s]")
    opt_parser.add_option("--step", type=int, help="Specify time step (otherwise world time step is used)")
    options, args = opt_parser.parse_args() #W
    print(options.speed)
    
if __name__ == '__main__':
    func()