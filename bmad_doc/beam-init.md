# Beam Initialization {#c:beam.init}

Some based programs track beams of particles instead of tracking
individual particles one-by-one. This can be useful for several reasons.
For example, tracking beams is useful when inter-bunch or intra-bunch
effects are to be simulated. Also tracking beams can simplify the
bookkeeping a program needs to do to calculate such quantities such as
the bunch size.

A based program has two standard ways to specify the initial
distribution of a beam. One is using a `beam_init_struct` `structure` ()
which holds parameters (for example, the beam emittances) from which a
distribution of particles can be constructed. The `beam_init_struct`
structure is explained in Section . The other way is to specify the
initial beam distribution via a file that has the individual particle
positions. This is covered in Section .

## Beam_Init_Struct Structure {#s:beam.init.struct}

The `beam_init_struct` `structure` () holds parameters which are used to
initialize a beam. The parameters of this structure, shown with their
default values, are:

::: example
type beam_init_struct character(200) :: position_file = \"\" !
Initialization file name. character distribution_type(3) ! \"ELLIPSE\",
\"KV\", \"GRID\", \"\" (default). type (ellipse_beam_init_struct)
ellipse(3) ! For ellipse beam distribution type (kv_beam_init_struct) KV
! For KV beam distribution type (grid_beam_init_struct) grid(3) ! For
grid beam distribution logical use_particle_start = F ! Use
particle_start instead of %center and %spin? character random_engine !
\"pseudo\" (default) or \"quasi\". character random_gauss_converter !
\"exact\" (default) or \"quick\". real center(6) = 0 ! Bench phase space
center offset. real t_offset = 0 ! Time offset. real center_jitter(6) =
0 ! Bunch center rms jitter real emit_jitter(2) = 0 ! %RMS a and b-mode
emittance jitter real sig_z_jitter = 0 ! bunch length RMS jitter real
sig_pz_jitter = 0 ! pz energy spread RMS jitter real random_sigma_cutoff
= -1 ! -1 =\> no cutoff used. integer n_particle = 0 ! Number of
simulated particles per bunch. logical renorm_center = T ! Renormalize
centroid? logical renorm_sigma = T ! Renormalize sigma? real spin(3) =
0, 0, 0 ! Spin (x, y, z) real a_norm_emit = 0 ! a-mode normalized
emittance (= $\beta\,\gamma\,\epsilon_{unnorm}$) real b_norm_emit = 0 !
b-mode normalized emittance (= $\beta\,\gamma\,\epsilon_{unnorm}$) real
a_emit = 0 ! a-mode unnormalized emittance (=
$\epsilon_{norm}/\beta\gamma$) real b_emit = 0 ! b-mode unnormalized
emittance (= $\epsilon_{norm}/\beta\gamma$) real dpz_dz = 0 !
Correlation of pz with longitudinal position. real dt_bunch = 0 ! Time
between bunches. real sig_z = 0 ! Z sigma in m. real sig_pz = 0 ! pz
sigma. real bunch_charge = 0 ! Charge in a bunch. integer n_bunch = 0 !
Number of bunches. integer ix_turn = 0 ! Turn index. character species =
\"\" ! Species. Default is reference particle. logical
full_6D_coupling_calc = F ! Use 6x6 1-turn matrix to match distribution?
logical use_t_coords = F ! If true, the distributions will be !
calculated using time coordinates (). logical use_z_as_t = F ! Only used
if use_t_coords = T: ! If True, the z coordinate stores the time. ! If
False, the z coordinate stores the s-position. end type
:::

Note: The `z` coordinate value given to particles of a bunch is with
respect to the nominal center of the bunch. Therefore, if there are
multiple bunches, and there is an RF cavity whose frequency is not
commensurate with the spacing between bunches, absolute time tracking ()
must be used.

%position_file

:   `%position_file` sets the name of the file to be read in containing
    the particle coordinates. Input from a file is triggered if
    not-blank. The format of the file is discussed in Section . %

%a_emit, %b_emit, %a_norm_emit, %b_norm_emit

:   Normalized and unnormalized emittances. Either `a_norm_emit` or
    `a_emit` may be set but not both. similarly, either `b_norm_emit` or
    `b_emit` may be set but not both.

    When simulating a ring, if any of these parameters is set negative,
    and if the based program being run has enabled it, the value of the
    parameter will be set the value as calculated from the lattice using
    synchrotron radiation integral formulas (). %

%bunch_charge

:   The `%bunch_charge` paramter sets the charge of a bunch. If reading
    from a file, the bunch charge will be set to the value of
    `%bunch_charge` except if `%bunch_charge` has a value of zero in
    which case the bunch charge as specified in the file is used. %

%center(6)

:   The `%center` parameter is used to offset the center position of a
    bunch. Exception: When `%use_particle_start` is set to True, the
    `particle_start` orbital values are used instead for the center
    position offset. See the description of `%use_particle_start` below
    for more details. %

%center_jitter, %emit_jitter, %sig_z_jitter, %sig_pz_jitter

:   These components can be used to provide a bunch-to-bunch random
    variation in the emittance and bunch center. %

%distribution_type(3)

:   The `%distribution_type(:)` array determines what algorithms are
    used to generate the particle distribution for a bunch. Note: If
    `%position_file` is not blank, the beam distribution will be read
    from the appropriate file and `%distribution_type` will be ignored.

    `%distribution_type(1)` sets the distribution type for the
    $(x, p_x)$ 2D phase space, etc. Possibilities for
    `%distribution_type(:)` are:

    ::: example
    \"\", or \"RAN_GAUSS\" ! Random distribution (default). \"ELLIPSE\"
    ! Ellipse distribution () \"KV\" ! Kapchinsky-Vladimirsky
    distribution () \"GRID\" ! Uniform distribution.
    :::

    Since the Kapchinsky-Vladimirsky distribution is for a 4D phase
    space, if the Kapchinsky-Vladimirsky distribution is used, `"KV"`
    must appear exactly twice in the `%distribution_type(:)` array.

    Unlike all other distribution types, the `GRID` distribution is
    independent of the Twiss parameters at the point of generation. For
    the non-`GRID` distributions, the distributions are adjusted if
    there is local $x$-$y$ coupling (). For lattices with a closed
    geometry, if `full_6D_coupling_calc` is set to `True`, the full
    6-dimensional coupling matrix is used. If `False`, which is the
    default, The 4-dimensional $\bfV$ matrix of is used.

    Note: The total number particles generated is the product of the
    individual distributions. For example:

    ::: example
    type (beam_init_struct) bi bi%distribution_type = ELLIPSE\",
    \"ELLIPSE\", \"GRID\" bi%ellipse(1)%n_ellipse = 4
    bi%ellipse(1)%part_per_ellipse = 8 bi%ellipse(2)%n_ellipse = 3
    bi%ellipse(2)%part_per_ellipse = 100 bi%grid(3)%n_x = 20
    bi%grid(3)%n_px = 30
    :::

    The total number of particles per bunch will be
    $32 \times 300 \times 600$. The exception is that when `RAN_GAUSS`
    is mixed with other distributions, the random distribution is
    overlaid with the other distributions instead of multiplying. For
    example:

    ::: example
    type (beam_init_struct) bi bi%distribution_type = RAN_GAUSS\",
    \"ELLIPSE\", \"GRID\" bi%ellipse(2)%n_ellipse = 3
    bi%ellipse(2)%part_per_ellipse = 100 bi%grid(3)%n_x = 20
    bi%grid(3)%n_px = 30
    :::

    Here the number of particle is $300 \times 600$. Notice that when
    `RAN_GAUSS` is mixed with other distributions, the value of
    `beam_init%n_particle` is ignored. %

%dPz_dz

:   Correlation between $p_z$ and $z$ phase space coordinates. %

%dt_bunch

:   Time between bunches %

%ellipse(3)

:   The `%ellipse(:)` array sets the parameters for the `ellipse`
    distribution (). Each component of this array looks like

    ::: example
    type ellipse_beam_init_struct integer part_per_ellipse ! number of
    particles per ellipse. integer n_ellipse ! number of ellipses. real
    sigma_cutoff ! sigma cutoff of the representation. end type
    :::

    \%

%full_6D_coupling_calc

:   If set `True`, coupling between the transverse and longitudinal
    modes is taken into account when calculating the beam distribution.
    The default `False` decouples the transverse and longitudinal
    calculations. %

%grid(3)

:   The `%grid` component of the `beam_init_struct` sets the parameters
    for a uniformly spaced grid of particles. The components of `%grid`
    are:

    ::: example
    type grid_beam_init_struct integer n_x ! number of columns. integer
    n_px ! number of rows. real x_min ! Lower x limit. real x_max !
    Upper x limit. real px_min ! Lower px limit. real px_max ! Upper px
    limit. end type
    :::

    \%

%ix_turn

:   Turn index. This affects how particle time is calculated. Particle
    time is calculated from phase space $z$ and $t_0$ via . $t_0$ is the
    reference time at the lattice element that the beam is initialized
    at (). For simulations where the beam is circulating in a ring over
    many turns, it may be desired to initialize the beam appropriate for
    some turn after the first turn. That is, set the particle time with
    the reference time $t_{0n}$ associated with the $n$ (set by
    `ix_turn`) turn at which the beam is being initialized at
    $$t_{0n} = t_0 + n \, t_{rev}$$ where $t_{rev}$ is the revolution
    time. %

%KV

:   The `%kv` component of the `beam_init_struct` sets the parameters
    for the Kapchinsky-Vladimirsky distribution (). The components of
    `%KV` are:

    ::: example
    type kv_beam_init_struct integer part_per_phi(2) ! number of
    particles per angle variable. integer n_I2 ! number of I2 real A ! A
    = I1/e end type
    :::

    \%

%n_bunch

:   The number of bunches in the beam is set by `n_bunch`. If reading
    the distribution from a file, if `%n_bunch` is zero, the number of
    bunches created is the number of defined in the file and if
    `%n_bunch` is not zero, the number created is `%n_bunch`. It is an
    error if `%n_bunch` is greater than the number of bunches defined in
    the file. If not reading from a file, if `%n_bunch` is zero, one
    bunch is created. %

%n_particle

:   Number of particles generated when the `%distribution_type` is
    `"RAN_GAUSS"`. Ignored for other distribution types. When reading
    the distribution from a file, if `%n_particle` is zero, the number
    of particles in a bunch will be the number of particles defined in
    the file. If `%n_particle` is non-zero when reading from a file, the
    number of particles in a bunch will be `%n_particle`. It is an error
    if `%n_particle` is non-zero and the number of particles defined in
    the file is less than `%n_particle`. %

%random_engine

:   This component sets the algorithm to use in generating a uniform
    distribution of random numbers in the interval \[0, 1\]. `"pseudo"`
    is a pseudo random number generator and \"quasi\" is a quasi random
    generator. \"quasi random\" is a misnomer in that the distribution
    generated is fairly uniform. %

%random_gauss_converter, %random_sigma_cutoff

:   To generate Gaussian random numbers, a conversion algorithm from the
    flat distribution generated according to `%random_engine` is needed.
    `%random_gauss_converter` selects the algorithm. The `"exact"`
    conversion uses an exact conversion. The `"quick"` method is
    somewhat faster than the `"exact"` method but not as accurate. With
    either conversion method, if `%random_sigma_cutoff` is set to a
    positive number, this limits the maximum sigma generated. %

%renorm_center, %renorm_sigma

:   If set to True, these components will ensure that the actual beam
    center and sigmas will correspond to the input values. Otherwise,
    there will be fluctuations due to the finite number of particles
    generated. %

%sig_pz, %sig_z

:   Longitudinal sigmas. `%sig_pz` is the fractional energy spread dE/E.
    This, along with `%dPz_dz` determine the longitudinal profile.

    When simulating a ring, if any of these parameters is set negative,
    and if the based program being run has enabled it, the value of the
    parameter will be set the value as calculated from the lattice using
    synchrotron radiation integral formulas (). %

%species

:   Name of the species tracked. If not set then the default tracking
    particle type is used. %

%spin

:   Particle spin in Cartesian $(x, y, z)$ coordinates. Only used when
    not reading in particle positions from a file. Also: When
    `%use_particle_start` is set to True, and not reading in particle
    positions form a file, the `particle_start` spin values are used
    instead. See the description of `%use_particle_start` below for more
    details. %

%use_particle_start

:   If `%use_particle_start` is set to `False` (the default) the center
    of the bunch is determined by the setting of the `%center`
    component. If set to `True`, the center is determined by the setting
    of:

    ::: example
    particle_start\[x\], particle_start\[px\], particle_start\[y\],
    particle_start\[py\], particle_start\[z\], particle_start\[pz\]
    :::

    If not reading from from a particle position file,
    `%use_particle_start` will affect the setting of the spin. If
    `%use_particle_start` is set to `False`, the initial spin
    orientation is determined by the setting of the `%spin` component.
    If set to `True`, the spin is determined by:

    ::: example
    particle_start\[spin_x\], particle_start\[spin_y\],
    particle_start\[spin_z\]
    :::

    See for details about the `particle_start` structure. Using the
    `particle_start` structure allows setting center and spin values in
    the lattice file rather than the setting of `%center` and `%spin` in
    the `beam_init_struct`. In this case, `%center` is a dependent
    parameter and will be set to the value of `particle_start`. %

%use_t_coords, %use_z_as_t

:   The problem with handling particle distributions at low energies is
    that when a particle's velocity is zero, the phase space
    $z$-coordinate is zero. What is needed here is to generate
    distributions in time and or in longitudinal $s$-position. To do
    this, two switches, `use_to_coords` and `use_z_as_t`, are used.

    If `use_t_coords` is True (default is False), the values in the
    distributions are taken as describing particles using "time
    coordinates" ().

    The `use_z_as_t` parameter is only used if `use_t_coords` is set to
    True. When `use_t_coords` is True, if `use_z_as_t` is also True, the
    value of the time (in seconds) assigned to a particle will be set
    equal to the `z`-coordinate (in meters) and a new `z`-coordinate
    will be calculated based upon . This is useful for describing the
    situation, for example, where particles may originate at a cathode
    at the same $s$-position, but different times. If `use_z_as_t` is
    False (the default), the $z$ coordinate is taken to describe the
    $s$-coordinate. This is useful for modeling particles that have the
    same time but different $s$ positions. In this case, particles are
    "born" inside the lattice element at the location of the particle's
    $s$-coordinate. To properly track the particles, the bunch will need
    to be tracked through the element where they are born with a
    tracking method that can handle inside particles. Currently the only
    such tracking method is `time_runge_kutta`. When the particles get
    to the end of the element, the particle positions are converted to
    standard $s$-coordinates.

    Since HDF5 beam files store complete information about a particle,
    `use_to_coords` and `use_z_as_t` are not needed and are ignored.

## File Based Beam Initialization {#s:beam.init.file}

A beam initialization file specifies the coordinates of all the
particles in a beam. If a based program uses a `beam_init_struct` () for
inputting initialization parameters, the file name for file based beam
initialization can be set using the `%position_file` component of the
structure. Also the bunch charge, bunch number, and number of particles
per bunch can be set in the `beam_init_struct`. Additionally, the bunch
centroid can be offset by setting `beam_init%center` and
`beam_init%center_jitter`.

There are two formats for the beam initialization file: ASCII and
binary. The binary file format for beam position storage is based on the
`HDF5` standard. More information is in chapter .

The new ASCII format describes a particle bunch with a `header` section
followed by a `table` of particle parameters. Multiple bunches that
comprise a beam can be specified by multiple header section / particle
parameter table pairs, one pair for each bunch. An example header
section:

::: example
\# The header field lines all start with a pound \"#\" sign. \# Any line
in the header field that does not start with a recognized parameter or
\# does not have an equal sign is ignored. \# my_param = 1.23 ! Custom
parameters can be defined and will be ignored by Bmad. \# species =
proton ! This parameter will be read by Bmad. \# spin = 1, 0, 0 ! and
this one too. \# The last line in the header field starts with \"#!\"
and defines the table columns. #! index x px y py \... etc\...
:::

The header section lines all start with a pound "\#" sign. The last line
in the header section must start with "#!" and this line defines the
particle parameter table columns. With the exception of the last line,
all header lines will be ignored except ones that begin with a
recognized parameter followed by an equal sign. Recognized parameters
are the components of the `coord_struct`, as documented in , that
describes individual particles. In addition, the following parameters
are recognized:

::: example
charge_tot ! Total bunch charge (including dead particles). s_position !
Longitudinal position of bunch. Can be used in place of \"s\". time !
Time particles of bunch are at. Can be used in place of \"t\".
:::

These parameters, if present, will be used to set the corresponding
particle parameter in all the particles of the corresponding bunch.
Possible `location` and `state` parameter settings are documented in .
The string equivalent to any setting is obtained by removing the
trailing "\$" from the variable name. For example, the variable `alive$`
which is a possible `state` setting becomes the string `alive`. Note:
`charge_tot` (total bunch charge) will superceed `charge` (the charge
per particle).

The particle parameter table follows the header section. Each row gives
the parameters for one particle. Not all particle parameters must be
specified. If a particular parameter is not present, its default value
will be used or, if present, the value given in the header section.
Column order is irrelavent and what determines what particle parameter
is associated with a given column is the last line of the header section
which starts with the characters `#!`. Except for vector parameters,
column names correspond to `coord_struct` parameters the same as in the
header section. For parameters that are vectors, the mapping from column
name to parameter is:

::: example
Column Name Corresponding coord_struct components ----------------------
----------------------------------------- x, px, y, py, z, pz vec(1),
\..., vec(6) spin_x, spin_y, spin_z spin(1), spin(2), spin(3) field_x,
field_y field(1), field(2) phase_x, phase_y phase(1), phase(2)
:::

Additionally, there can be (but is not required) an `index` column
giving the particle index in the array of particles of a bunch. This
column is ignored so, for example, values in the first line of the table
will always be used to set the particle with index 1 independent of the
value given in the index column. This behavior is implemented so that a
beam file can be edited to add or remove particles without worrying
about reindexing.

Note: Example beam files (both ASCII and HDF5) can be generated using
the program in the Bmad Distribution `code_examples/beam_track_example`.

Note: Beam files can be converted between ASCII and HDF5 binary using
the program in the Bmad Distribution
`util_programs/beam_file_translate_format`.

### Old Beam ASCII Format

There is an old ASCII format that is still accepted by but is deprecated
and should be avoided. The format is:

::: example
\<ix_ele\> ! Lattice element index. This is ignored. \<n_bunch\> !
Number of bunches. \<n_particle\> ! Number of particles per bunch to use
\[bunch loop: ib = 1 to n_bunch\] BEGIN_BUNCH ! Marker to mark the
beginning of a bunch specification block. \<species_name\> ! Species of
particle \<charge_tot\> ! Total charge of bunch (alive + dead). 0 =\>
Use \<macro_charge\>. \<z_center\> ! z position at center of bunch.
\<t_center\> ! t position at center of bunch. \[particle loop: Stop when
END_BUNCH marker found\] \<x\> \<px\> \<y\> \<py\> \<z\> \<pz\>
\<macro_charge\> \<state\> \<spin_x\> \<spin_y\> \<spin_z\> \[end
particle loop\] END_BUNCH ! Marker to mark the end of the bunch
specification block \[end bunch loop\]
:::

Example:

::: example
0 ! ix_ele 1 ! n_bunch 25000 ! n_particle BEGIN_BUNCH POSITRON 3.2E-9 !
charge_tot 0.0 ! z_center 0.0 ! t_center -6.5E-3 9.6E-3 -1.9E-2 8.8E-3
2.2E-2 -2.4E-2 1.2E-13 1 1.0 0.0 0.0 8.5E-3 5.5E-3 4.0E-2 -1.9E-2
-4.9E-3 2.1E-2 1.2E-13 1 1.0 0.0 0.0 1.1E-2 -1.9E-2 -2.5E-2 1.0E-2
-1.8E-2 -7.1E-3 1.2E-13 1 1.0 0.0 0.0 -3.4E-2 -2.7E-3 -4.1E-3 1.3E-2
1.3E-2 1.0E-2 1.2E-13 1 1.0 0.0 0.0 6.8E-3 -4.5E-3 2.5E-3 1.4E-2 -2.3E-3
7.3E-2 1.2E-13 1 1.0 0.0 0.0 1.2E-2 -9.8E-3 1.7E-3 6.4E-3 -9.8E-3
-7.2E-2 1.2E-13 1 1.0 0.0 0.0 1.1E-2 -3.5E-4 1.2E-2 1.8E-2 5.4E-3 1.4E-2
1.2E-13 1 1.0 0.0 0.0 \... etc. \... END_BUNCH
:::

The first line of the file gives `<ix_ele>`, the index of the lattice
element at which the distribution was created. This is ignored when the
file is Read.

The second line gives `<n_bunch>`, the number of bunches. This can be
overridden by a non-zero setting of `beam_init%n_bunch`.

The third line gives `<n_particle>` the number of particles in a bunch.
The actual number rows specifying particle coordinates may be more then
`<n_particle>`. In this case, particles will be discarded so that the
beam has `<n_particle>` particles. The setting of `<n_particle>` can be
overridden by a non-zero setting of `beam_init%n_particle`.

After this, there are `<n_bunch>` blocks of data, one for each bunch.
Each one of these blocks starts with a `BEGIN_BUNCH` line to mark the
beginning of the block and ends with a `END_BUNCH` marker line. In
between, the first four lines give the `<species_name>` name,
`<charge_tot>`, `<z_center>`, and `<t_center>` values. The
`<species_name>` name may be one of:

::: example
positron ! default electron proton antiproton muon antimuon photon
:::

The lines following the `<t_center>` line specify particle coordinates.
One line for each particle. Only the first six numbers, which are the
phase space coordinates, need to be specified for each particle. If the
`<macro_charge>` column is not present, or is zero, it defaults to
`<charge_tot>/<n_particle>`.

The `<state>` parameter indicates whether a particle is alive or dead.
Values are

::: example
1 ! Alive 2-7 ! Dead 8 ! Pre-born
:::

The `pre-born` state indicates that the particle is waiting to be
emitted from the cathode of an electron gun ().

The particle spin is specified by $x$, $y$ and $z$ components.

Each particle has an associated `<macro_charge>`. If `<charge_tot>` is
set to a non-zero value, the charge of all the particles will be scaled
by a factor to make the total macro charge equal to `<charge_tot>`. The
macro charge is ignored in tracking. The charge of the particle used in
tracking is the charge as calculated for the particle species. On the
other hand, the macro charge is used to calculate such things as the
total charge in a particular region or the field produced by a particle.
That is, the macro charge acts as a weighting factor for a particle when
the particle's field or the particle's effect on other particles is
calculated.

When the particle coordinates are read in the centroid will be shifted
by the setting of `beam_init%center` (unless
`beam_init%use_particle_start` is set True) and
`beam_init%center_jitter`.
