#!/bin/bash

rm -rf results/*
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n supervised_no_domain_class -d widar3 -d_t ampphase -t in-domain -do_t random
python experiment_automation.py -m_n supervised_no_domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n supervised_no_domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t position
python experiment_automation.py -m_n supervised_no_domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t orientation

python experiment_automation.py -m_n supervised_no_domain_class -d signfi -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n supervised_no_domain_class -d signfi -d_t ampphase -t domain-leave-out -do_t environment

mkdir results/supervised_no_domain_class
mv results/*.csv results/supervised_no_domain_class
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n no_domain_class -d widar3 -d_t ampphase -t in-domain -do_t random
python experiment_automation.py -m_n no_domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n no_domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t position
python experiment_automation.py -m_n no_domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t orientation

python experiment_automation.py -m_n no_domain_class -d signfi -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n no_domain_class -d signfi -d_t ampphase -t domain-leave-out -do_t environment

mkdir results/no_domain_class
mv results/*.csv results/no_domain_class
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n domain_class -d widar3 -d_t ampphase -t in-domain -do_t random
python experiment_automation.py -m_n domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t position
python experiment_automation.py -m_n domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t orientation

python experiment_automation.py -m_n domain_class -d signfi -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n domain_class -d signfi -d_t ampphase -t domain-leave-out -do_t environment

mkdir results/domain_class
mv results/*.csv results/domain_class
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n multilabel_domain_class -d widar3 -d_t ampphase -t in-domain -do_t random
python experiment_automation.py -m_n multilabel_domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n multilabel_domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t position
python experiment_automation.py -m_n multilabel_domain_class -d widar3 -d_t ampphase -t domain-leave-out -do_t orientation

python experiment_automation.py -m_n multilabel_domain_class -d signfi -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n multilabel_domain_class -d signfi -d_t ampphase -t domain-leave-out -do_t environment

mkdir results/multilabel_domain_class
mv results/*.csv results/multilabel_domain_class
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n multilabel_domain_class_2 -d widar3 -d_t ampphase -t in-domain -do_t random
python experiment_automation.py -m_n multilabel_domain_class_2 -d widar3 -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n multilabel_domain_class_2 -d widar3 -d_t ampphase -t domain-leave-out -do_t position
python experiment_automation.py -m_n multilabel_domain_class_2 -d widar3 -d_t ampphase -t domain-leave-out -do_t orientation

python experiment_automation.py -m_n multilabel_domain_class_2 -d signfi -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n multilabel_domain_class_2 -d signfi -d_t ampphase -t domain-leave-out -do_t environment

mkdir results/multilabel_domain_class_2
mv results/*.csv results/multilabel_domain_class_2
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n domain_aware_batch -d widar3 -d_t ampphase -t in-domain -do_t random
python experiment_automation.py -m_n domain_aware_batch -d widar3 -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n domain_aware_batch -d widar3 -d_t ampphase -t domain-leave-out -do_t position
python experiment_automation.py -m_n domain_aware_batch -d widar3 -d_t ampphase -t domain-leave-out -do_t orientation

python experiment_automation.py -m_n domain_aware_batch -d signfi -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n domain_aware_batch -d signfi -d_t ampphase -t domain-leave-out -do_t environment

mkdir results/domain_aware_batch
mv results/*.csv results/domain_aware_batch
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n domain_aware_filter_denom -d widar3 -d_t ampphase -t in-domain -do_t random
python experiment_automation.py -m_n domain_aware_filter_denom -d widar3 -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n domain_aware_filter_denom -d widar3 -d_t ampphase -t domain-leave-out -do_t position
python experiment_automation.py -m_n domain_aware_filter_denom -d widar3 -d_t ampphase -t domain-leave-out -do_t orientation

python experiment_automation.py -m_n domain_aware_filter_denom -d signfi -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n domain_aware_filter_denom -d signfi -d_t ampphase -t domain-leave-out -do_t environment

mkdir results/domain_aware_filter_denom
mv results/*.csv results/domain_aware_filter_denom
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n domain_aware_batch_alt_flow -d widar3 -d_t ampphase -t in-domain -do_t random
python experiment_automation.py -m_n domain_aware_batch_alt_flow -d widar3 -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n domain_aware_batch_alt_flow -d widar3 -d_t ampphase -t domain-leave-out -do_t position
python experiment_automation.py -m_n domain_aware_batch_alt_flow -d widar3 -d_t ampphase -t domain-leave-out -do_t orientation

python experiment_automation.py -m_n domain_aware_batch_alt_flow -d signfi -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n domain_aware_batch_alt_flow -d signfi -d_t ampphase -t domain-leave-out -do_t environment

mkdir results/domain_aware_batch_alt_flow
mv results/*.csv results/domain_aware_batch_alt_flow
rm -rf Streams/*
rm -rf tmp/*

python experiment_automation.py -m_n domain_aware_filter_denom_alt_flow -d widar3 -d_t ampphase -t in-domain -do_t random
python experiment_automation.py -m_n domain_aware_filter_denom_alt_flow -d widar3 -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n domain_aware_filter_denom_alt_flow -d widar3 -d_t ampphase -t domain-leave-out -do_t position
python experiment_automation.py -m_n domain_aware_filter_denom_alt_flow -d widar3 -d_t ampphase -t domain-leave-out -do_t orientation

python experiment_automation.py -m_n domain_aware_filter_denom_alt_flow -d signfi -d_t ampphase -t domain-leave-out -do_t user
python experiment_automation.py -m_n domain_aware_filter_denom_alt_flow -d signfi -d_t ampphase -t domain-leave-out -do_t environment

mkdir results/domain_aware_filter_denom_alt_flow
mv results/*.csv results/domain_aware_filter_denom_alt_flow
rm -rf Streams/*
rm -rf tmp/*
