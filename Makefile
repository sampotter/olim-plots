MIN_2D_POWER = 3
MAX_2D_POWER = 4

MIN_3D_POWER = 3
MAX_3D_POWER = 4

DEPS_2D = common2.py slow2.py
DEPS_3D = common3.py slow3.py

speed_comparison.npz: collect_speed_comparison_data.py
	./collect_speed_comparison_data.py

time_vs_error_2d.eps size_vs_error_2d.eps: make_2d_time_vs_error_plots.py $(DEPS_2D)
	./make_2d_time_vs_error_plots.py \
		--min_2d_power=$(MIN_2D_POWER) --max_2d_power=$(MAX_2D_POWER)

size_vs_error_3d.eps: make_3d_size_vs_error_plots.py $(DEPS_3D)
	./make_3d_size_vs_error_plots.py \
		--min_3d_power=$(MIN_3D_POWER) --max_3d_power=$(MAX_3D_POWER)

time_vs_error_3d.eps: make_3d_time_vs_error_plots.py $(DEPS_3D)
	./make_3d_time_vs_error_plots.py \
		--min_3d_power=$(MIN_3D_POWER) --max_3d_power=$(MAX_3D_POWER)

factoring_example.eps: make_factoring_example_plot.py slow2.py slow3.py
	./make_factoring_example_plot.py \
		--min_2d_power=$(MIN_2D_POWER) --max_2d_power=$(MAX_2D_POWER) \
		--min_3d_power=$(MIN_3D_POWER) --max_3d_power=$(MAX_3D_POWER)

intro.eps: make_intro_plot.py common3.py
	./make_intro_plot.py \
		--min_3d_power=$(MIN_3D_POWER) --max_3d_power=$(MAX_3D_POWER)

qv_plots.eps: make_qv_plots.py common2.py common3.py
	./make_qv_plots.py \
		--min_2d_power=$(MIN_2D_POWER) --max_2d_power=$(MAX_2D_POWER) \
		--min_3d_power=$(MIN_3D_POWER) --max_3d_power=$(MAX_3D_POWER)

speed_comparison.eps: make_speed_comparison_plot.py
	./make_speed_comparison_plot.py


