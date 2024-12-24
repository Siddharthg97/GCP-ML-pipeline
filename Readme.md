### Steps to create GCP ML pipeline

1) Settings.yml file - contains model configuration & data location
2) ML pipeline scripts - 
3) Utils module  - contains function definitions
4) base flow image 
5) ML flow image
6) Pipeline_utils
7) Decryption


To create ML flow pipeline we need containerized applications conatining -python configurations, 




**Notes** 
We can create our own argument parser
It is basucally a container that contains all the argument we want to pass from commmand line.
https://docs.python.org/3/library/argparse.html
https://www.youtube.com/watch?v=FsAPt_9Bf3U
    parser = argparse.ArgumentParser()
    parser.add_argument("--COMMIT_ID", required=True, type=str)
    parser.add_argument("--BRANCH", required=True, type=str)
    parser.add_argument("--is_prod", required=False, type=lambda x: (str(x).lower() == 'true'))
    sys.args = [
        "--COMMIT_ID", "1234",
        "--BRANCH", "dev",
        "--is_prod", False,
    ]
    args = parser.parse_args(sys.args)




   
