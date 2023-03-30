import enum


class EnumConstant(enum.Enum):
    def __str__(self):
        return self.value


class MosaicArgs(EnumConstant):
    CATSDOGS_MOSAIC = 'catsdogs_mosaic'
    MAME_MOSAIC ='mame_mosaic'
    MIT67_MOSAIC = 'mit670_mosaic'


class XmethodArgs(EnumConstant):
    LRP = 'lrp'
    GRADCAM = 'gradcam'
    INTGRAD = 'intgrad'
    LIME = 'lime'


class ArchArgs(EnumConstant):
    RESNET18 = 'resnet18'
    VGG16 = 'vgg16'
