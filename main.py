from make_paper_plots import *

if __name__ == "__main__":
    testdir = '/share/hypatia/sbird/Lya_Boss/hires_knots_test'
    emudir = '/share/hypatia/sbird/Lya_Boss/hires_knots'

    test_knot_plots(testdir=testdir, emudir=emudir, plotname="One_kf", kf_bin_nums=np.arange(1))
    #test_knot_plots(testdir='/Users/kwame/Simulations/Lya_Boss/hires_knots_test',emudir='/Users/kwame/Simulations/Lya_Boss/hires_knots')
