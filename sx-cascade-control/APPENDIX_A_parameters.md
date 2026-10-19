# Appendix A, parameters: every number, with its origin

Rendered from `params/parameters.toml` by `notebooks/12_parameter_table.py`. The note on each
line is the one in the file: *measured* means it is in a cited source or fitted from one by a
script in this repository, *chosen* means picked by me with the rule stated, *derived* means
arithmetic on the others. Sources are abbreviated as in the file header (IDAHO = Lyon 2016;
YIN18 = Yin et al. 2018; ECJTU = US 11,214,850 B2; CARESTER = Rollat et al. 2024; AE = American
Elements product page).

| section | parameter | value | origin |
| --- | --- | --- | --- |
| meta | `schema` | `1` |  |
| meta | `notes` | `"See ARCHITECTURE.md decision 4. Run-specific choices (flows, stage counts, sensors) live in configs/, not here."` |  |
| components | `names` | `["LaCe", "Pr", "Nd", "SmPlus"]` |  |
| components.molar_mass | `LaCe` | `139.7` | chosen: 1/3 La (138.91) + 2/3 Ce (140.12), typical monazite La:Ce Why: monazite usually carries about twice as much Ce as La (textbook) |
| components.molar_mass | `Pr` | `140.91` | measured: periodic table |
| components.molar_mass | `Nd` | `144.24` | measured |
| components.molar_mass | `SmPlus` | `150.36` | chosen: Sm stands for the whole heavy lump |
| components.oxide_mass_per_metal | `LaCe` | `169.0` | chosen: 1/3 La2O3 (162.9 per La) + 2/3 CeO2 (172.1 per Ce) |
| components.oxide_mass_per_metal | `Pr` | `170.2` | derived: Pr6O11 1021.4 / 6 |
| components.oxide_mass_per_metal | `Nd` | `168.3` | derived: Nd2O3 336.5 / 2 |
| components.oxide_mass_per_metal | `SmPlus` | `174.4` | derived: Sm2O3 348.7 / 2 |
| extractant | `name` | `"PC88A"` | = P507 = HEH(EHP), the industrial acidic extractant (CARESTER, IDAHO) |
| extractant | `diluent` | `"kerosene"` |  |
| extractant | `concentration_monomer` | `1.0` | mol/L, measured: IDAHO ran 1 M PC88A in Isopar-L throughout |
| extractant | `max_loading_fraction` | `0.85` | chosen: the design may load the organic to 85 % of its theoretical capacity (dimer_total / ligands_per_metal) in any stage. Without it the reagent-minimizing optimizer runs every extraction and scrub stage at 95-99 % and D (which goes as L_free^3) becomes so steep that a plant a few percent off the model is outside what the controller's linearization can describe (ARCHITECTURE.md, 22). Judgement call: plants do run the scrub organic close to full, since that's what makes the scrub work, but 85 % isn't from plant data. Check: the design's saponification (0.4285) is about what an organic at this cap gives off (0.85 x 0.5 = 0.425). Loading isotherms to saturation on the real organic would settle it. |
| extractant | `dimer_fraction` | `1.0` | chosen: treat all extractant as dimer (H2L2), IDAHO Eq. 4 |
| extractant | `ligands_per_metal` | `3` | measured: RE3+ + 3 H2L2 -> RE(HL2)3 + 3 H+  (CARESTER eq. 1, IDAHO Eq. 4) |
| equilibrium | `law` | `"idaho_eq4"` |  |
| equilibrium | `acid_exponent` | `3.0` | chosen (see above); a setting, so n = 3.9/4.5 can be tried |
| equilibrium | `D_max` | `1e4` | chosen: smooth cap D -> D / (1 + D / D_max). With nearly empty organic and acid near its floor the law gives D ~ 1e6, which only makes the equations stiff; at D = 1e4 extraction is complete anyway (99.99 % at O/A 1). Loaded stages sit at D ~ 1e2, untouched. 0 = off. |
| equilibrium | `acid_floor_mol_per_L` | `1e-3` | chosen: D = K L^3 / (h^n + floor^n). Below ~pH 3 the law stops applying (hydrolysis, hydroxide precipitation) and D would run to infinity; the floor keeps it finite. At the fitted data (h >= 0.04) it changes D by < 2e-5. Judgement call: pH 3 is where the data stop; light lanthanide hydroxides come down nearer pH 6-7 (textbook). At the ndpr_plant point no stage is above pH 1.8, so the floor isn't reached. |
| equilibrium.K | `Nd` | `0.0321` | measured: fit, IDAHO Table 4 (10 points) |
| equilibrium.K | `Pr` | `0.0212` | measured: fit, IDAHO Table 5 (10 points). beta Nd/Pr = 1.51 check: IDAHO Table 6 mixed metal measured 1.57-1.59; YIN18 industrial 1.3-1.5 |
| equilibrium.K | `LaCe` | `0.0106` | chosen: K_Pr / 2.0. YIN18 quotes industrial beta Pr/Ce = 1.8-2.2; midpoint 2.0. La extracts less still (beta Ce/La 3-5, YIN18), so using Ce's value for the whole lump is the conservative side: it makes La+Ce harder to separate from Pr. Load-bearing for the main Ce/Pr cut (the NdPr product); not for the Pr/Nd validation split. Judgement call: lower Pr/Ce values for P507 in chloride also appear in the literature (not checked against a project source). A Fenske estimate for this cut needs about 14 stages at total reflux at beta 2.0 and about 20 at 1.6, so the stage count moves with this number. A batch test on the plant feed at scrub-section strength would settle it. |
| equilibrium.K | `SmPlus` | `0.321` | chosen: K_Nd * 10. CARESTER Fig. 1 (log plot, HEH(EHP) curve) reads Sm/Nd ~ 14; handbook values commonly quoted ~6-7 (not verified). 10 sits between. Not load-bearing: Sm+ is removed upstream in the NdPr runs, and where present it loads first whatever the exact value. |
| kinetics | `mixer_time_to_equilibrium_h` | `0.05` | measured: IDAHO, 3 min |
| kinetics | `approach_fraction` | `0.95` | chosen: what "achieve equilibrium" is taken to mean Judgement call: 0.99 would give k five times higher and stages near 96 % at the run files' flows; 0.90 the reverse. REPORT section 3 shows how far that moves the stage count. |
| stage_geometry | `lab_mixer_volume_L` | `0.120` | measured: IDAHO Table 3 |
| stage_geometry | `lab_settler_volume_L` | `0.480` | measured: IDAHO Table 3 |
| stage_geometry | `lab_total_throughput_L_per_h` | `2.4` | measured: IDAHO, 40 mL/min, the throughput at which stages reach equilibrium |
| stage_geometry | `scale_factor` | `250` | chosen, rule above. The run files' design flows are higher than 600 L/h through each mixer, so their stages are less efficient than the lab's (see [kinetics]) |
| stage_geometry | `mixer_volume_L` | `30.0` | derived: 0.120 * 250 |
| stage_geometry | `settler_volume_L` | `120.0` | derived: 0.480 * 250 |
| stage_geometry | `settler_height_m` | `0.5` | chosen: round number; only ratios to it matter |
| stage_geometry | `settler_area_m2` | `0.24` | derived: 120 L / 0.5 m |
| stage_geometry | `organic_weir_height_m` | `0.45` | chosen: 90 % of height |
| stage_geometry | `interface_setpoint_fraction` | `0.5` | chosen: mid-window |
| stage_geometry | `interface_window_fraction` | `[0.3, 0.7]` | chosen: where the MPC must keep it |
| stage_geometry | `jackleg_time_constant_h` | `0.1` | chosen: 6 min; swept in the disturbance study |
| densities | `organic_unloaded_g_per_mL` | `0.85` | chosen: 1 M PC88A (~0.32 L/L at ~0.95) in kerosene (~0.80) |
| densities | `aqueous_water_g_per_mL` | `0.997` | measured: water, 25 C |
| densities | `apparent_molal_volume_RECl3_mL_per_mol` | `15.0` | chosen: order of magnitude for RECl3 at 0.1-1 m Judgement call: free HCl and the NaCl from saponification aren't counted in rho_aq (see chemistry.rho_aqueous) |
| densities | `chloride_molar_mass` | `35.45` | measured: periodic table |
| feed_defaults | `deposit` | `"eneabba"` |  |
| feed_defaults | `total_RE_g_per_L` | `50.0` | chosen: IDAHO ran 50 g/L didymium feeds; same order as plant liquors Judgement call: g/L of metal (about 60 as oxide). Feeds to P507 plants are often stronger than this (practice, not checked against a project source); a stronger feed changes the flows and O/A more than the chemistry. |
| feed_defaults | `acid_mol_per_L` | `0.05` | chosen: inside IDAHO's tested range 0.01-0.3 M |
| feed_defaults | `removed_upstream` | `["LaCe", "SmPlus"]` | chosen: the default is the feed of the Pr/Nd validation split, didymium: Eneabba Pr:Nd = 26:74 (mol), the same 25/75 feed the IDAHO circuit ran. The main case, the NdPr product (cut "LaCe"), sets its own feed in configs/ndpr_*.toml: Sm+ removed upstream. |
| targets | `product_purity_fraction` | `0.99` | chosen: IDAHO ">99 % purity neodymium ... magnet applications"; AE lists didymium oxide from 99 % grade |
| targets | `recovery_fraction` | `0.95` | chosen: round number for "good recovery"; the stage-count study reports the whole purity-recovery curve anyway Judgement call: NdPr carries most of the value of a light rare-earth feed, so a plant would likely push recovery higher; the stage-count curve shows what that costs. |
| targets | `design_purity_fraction` | `0.995` | chosen: the design point and the controller's purity target sit half a point above spec, as plants run |
| design | `net_product_L_per_h` | `300.0` | chosen: strip product leaving the circuit; with a reflux scrub the strip flow is this plus the scrub flow |
| design | `pump_max_L_per_h` | `4700.0` | chosen: the organic pump's ceiling (the controller's bound) |
| design | `pump_headroom_fraction` | `0.85` | chosen: the design uses at most this share of the pump, so a controller can add organic (ARCHITECTURE.md, 19) |
| design | `organic_min_L_per_h` | `600.0` | chosen: lower bound on organic flow for design and control |
| anchors | `idaho_total_stages` | `18` | measured: IDAHO 4.2.3 |
| anchors | `idaho_scrub_stages` | `12` | measured |
| anchors | `idaho_product_purity` | `0.997` | measured: 99.7 % Nd from 25/75 Pr/Nd feed |
| anchors | `idaho_hours_to_steady_state` | `30` | measured |
| anchors | `ecjtu_total_stages` | `60` | measured: ECJTU full text |
| anchors | `ecjtu_extraction_stages` | `26` | measured |
| anchors | `ecjtu_scrub_stages` | `34` | measured ("washing section") |
| anchors | `ecjtu_feed_L_per_min` | `10` | measured |
| sensors | `flow_noise_relative` | `0.01` | chosen: a good magnetic flowmeter, ~1 % |
| sensors | `acid_noise_relative` | `0.05` | chosen: pH to +-0.02 units is ~5 % in [H+] Judgement call: this takes the reading as h itself. A pH probe reads activity, and in 0.8 M RECl3 or 2 M HCl that can differ noticeably from -log h; below pH 0 a probe is no use and plants titrate free acid. A plant calibration settles it. |
| sensors | `density_noise_g_per_mL` | `0.002` | chosen: Coriolis / vibrating-tube density meter |
| sensors | `interface_noise_m` | `0.002` | chosen: the interface-level-vision project read to ~2 mm |
| sensors | `onstream_noise_relative` | `0.02` | chosen: XRF / UV-Vis on-stream analyser, ~2 % of reading |
| sensors | `onstream_interval_h` | `0.25` | chosen: one analysis per quarter hour |
| sensors | `onstream_delay_h` | `0.1` | chosen: sample line and analysis, ~6 min |
| sensors | `lab_noise_relative` | `0.005` | chosen: ICP in a lab, ~0.5 % |
| sensors | `lab_interval_h` | `2.0` | chosen: a shift takes a sample every two hours |
| sensors | `lab_delay_h` | `4.0` | chosen: result four hours later |
| sensors | `process_noise_relative` | `0.01` | chosen: how much the estimator distrusts its own model per step |

Counted by label where the note starts with one: 19 measured, 40 chosen, 6 derived (entries whose note starts otherwise are not counted).
