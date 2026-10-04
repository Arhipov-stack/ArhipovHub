"""All dimensions in mm. Change these first if your speaker measures differently."""

# Yandex Station Midi (YNDX-00054): rounded cube 96 x 96 x 110 (W x D x H),
# USB-C on the back, LED display on the front, touch panel on top.
SPEAKER_W = 96.0
SPEAKER_D = 96.0
SPEAKER_H = 110.0

CLEARANCE = 1.5          # per side between speaker and pocket walls
POCKET_R = 8.0           # pocket corner radius; smaller than the speaker's, so it always fits

WALL = 5.0               # cradle wall
FLOOR = 6.0              # cradle floor under the speaker
RIM_Z = 92.0             # height of the side/back cradle walls above the floor

# neck joint (square, so the head can't turn)
PEG = 34.0               # peg side on the base
PEG_LEN = 20.0
SOCKET_GAP = 0.3         # per side
SOCKET_DEPTH = 22.0

# horn joint
HORN_PEG_D = 10.0
HORN_PEG_LEN = 10.0
HORN_HOLE_GAP = 0.2      # per side

PX = SPEAKER_W / 2 + CLEARANCE   # pocket half-width
PY = SPEAKER_D / 2 + CLEARANCE
