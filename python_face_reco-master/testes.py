from pyfirmata import Arduino,util, STRING_DATA
placa = Arduino("/dev/ttyUSB0")
placa.send_sysex(STRING_DATA,util.str_to_two_byte_iter('Just Do It '))

def msg( text ):
    if text:
        placa.send_sysex(STRING_DATA,util.str_to_two_byte_iter(text))

