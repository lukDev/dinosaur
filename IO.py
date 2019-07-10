class IO:
    heldKeys = set()

    @staticmethod
    def keyPressed(key):
        IO.heldKeys.add(key)

    @staticmethod
    def keyReleased(key):
        IO.heldKeys.remove(key)

    @staticmethod
    def isKeyHeld(key):
        return key in IO.heldKeys
