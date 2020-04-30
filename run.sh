#python main.py train --model=VGGish --features=log_mel --validate --manually_verified_only --shuffle \
    #--verbose
#sleep 600
#python main.py train --model=VGGish --features=mfcc --validate --manually_verified_only --shuffle \
    #--verbose
#sleep 600
#python main.py train --model=VGGish --features=chroma --validate --manually_verified_only --shuffle \
    #--verbose
#sleep 600

python plot.py train --model=VGGish --features=log_mel --validated --manually_verified_only --latex --verbose
#python plot.py train --model=VGGish --features=mfcc --validated --manually_verified_only --verbose
#python plot.py train --model=VGGish --features=chroma --validated --manually_verified_only --verbose

#python main.py validation --model=VGGish --features=log_mel --epoch=15 --validated \
    #--manually_verified_only --shuffle --verbose
#python plot.py validation --model=VGGish --features=log_mel --epoch=15 --validated \
    #--manually_verified_only --verbose

#python main.py test --model=VGGish --features=log_mel --epoch=15 --validated \
    #--manually_verified_only --verbose
#python plot.py test --model=VGGish --features=log_mel --epoch=15 --validated \
    #--manually_verified_only --verbose