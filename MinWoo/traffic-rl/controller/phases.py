"""Connection indices are explicitly assigned by build_network.py."""
PHASE_NAMES = ("NS straight", "NS left", "EW straight", "EW left")
LINK_DIRECTIONS = ("N", "E", "S", "W")
ACTIVE_LINKS = ((0, 1, 2, 8, 9, 10), (3, 11), (4, 5, 6, 12, 13, 14), (7, 15))
GREEN_STATES = tuple("".join("G" if i in active else "r" for i in range(16)) for active in ACTIVE_LINKS)
YELLOW_STATES = tuple(state.replace("G", "y") for state in GREEN_STATES)
ALL_RED = "r" * 16
