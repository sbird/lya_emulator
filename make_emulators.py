"""Script containing details of the emulators I ran."""

import coarse_grid
from quadratic_emulator import QuadraticEmulator

#Initial narrow limits
param_limits_3 = np.array([[0.9, 0.99], [1.4e-09, 2.6e-09], [-0.7, 0.1], [0.6, 1.4], [0.65, 0.75]])
emu = coarse_grid.Emulator("/home/spb/data/Lya_Boss_3/hires_s8", param_limits = param_limits_3)
emu.gen_simulations(21)
quad = QuadraticEmulator("/home/spb/data/Lya_Boss_3/hires_s8_quad", param_limits = param_limits_3)
quad.gen_simulations(21)
#Test simulations
emu_test = coarse_grid.Emulator("/home/spb/data/Lya_Boss_3/hires_s8_test", param_limits = param_limits_3)
emu_test.gen_simulations(6)

#Slightly wider limits
param_limits_4 = np.array([[0.8, 0.995], [1.2e-09, 2.6e-09], [-0.7, 0.1], [0.4, 1.4], [0.65, 0.75]])
emu = coarse_grid.Emulator("/home/spb/data/Lya_Boss_4/hires_s8", param_limits = param_limits_4)
emu.gen_simulations(21)
quad = QuadraticEmulator("/home/spb/data/Lya_Boss_4/hires_s8_quad", param_limits = param_limits_4)
quad.gen_simulations(21)
