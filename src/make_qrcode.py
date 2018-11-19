# -*- coding: utf-8 -*-
#!/usr/bin/env python
import qrcode

image = qrcode.make("what you want to say")
image.save('scanMe.png')
