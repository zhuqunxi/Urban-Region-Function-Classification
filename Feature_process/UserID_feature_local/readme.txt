服务器
@erisone.partners.org

qz062
Ab643588
Ab84607120

python 3设置
module load anaconda/3-5.0.1
pip install --user lightgbm
python -V
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

cd UPO/Mackey_UPO_RC_system

bsub -q big-multi -n 4 -R 'rusage[mem=20000]' < Good_Job_normal.lsf

bsub -q medium < Good_Job_1.lsf

bsub -q normal < my_test.lsf
bsub -q normal < Good_Job_1.lsf
bsub -q normal < Good_Job_2.lsf
bsub -q normal < Good_Job_3.lsf
bsub -q normal < Good_Job_4.lsf

bsub -q big-multi -n 4 -R 'rusage[mem=20000]' < Good_Job_1.lsf 
bsub -q big-multi -n 4 -R 'rusage[mem=20000]' < Good_Job_2.lsf 
bsub -q big-multi -n 4 -R 'rusage[mem=20000]' < Good_Job_3.lsf 
bsub -q big-multi -n 4 -R 'rusage[mem=20000]' < Good_Job_4.lsf 

bsub -q big-multi -n 4 -R 'rusage[mem=100000]' < Good_Job_1.lsf 
bsub -q big-multi -n 4 -R 'rusage[mem=100000]' < Good_Job_2.lsf 
bsub -q big-multi -n 4 -R 'rusage[mem=100000]' < Good_Job_3.lsf 
bsub -q big-multi -n 4 -R 'rusage[mem=100000]' < Good_Job_4.lsf 

bsub -q big-multi -n 8 -R 'rusage[mem=100000]' < Good_Job_1.lsf 
bsub -q big-multi -n 8 -R 'rusage[mem=100000]' < Good_Job_2.lsf 
bsub -q big-multi -n 8 -R 'rusage[mem=100000]' < Good_Job_3.lsf 
bsub -q big-multi -n 8 -R 'rusage[mem=100000]' < Good_Job_4.lsf 

bsub -q big-multi -R 'rusage[mem=100000]' < Good_Job_1.lsf 
bsub -q big-multi -R 'rusage[mem=100000]' < Good_Job_2.lsf 
bsub -q big-multi -R 'rusage[mem=100000]' < Good_Job_3.lsf 
bsub -q big-multi -R 'rusage[mem=100000]' < Good_Job_4.lsf 


bjobs -a
bkill job_id
bpeek

cd UPO/Baidu_competition/code/visit/MGH/Code_UserID_feature/Normal_global