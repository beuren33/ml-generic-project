import sys

def erro_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    erro_menssage = "Erro no arquivo: [{0}] na linha: [{1}], erro: [{2}]".format()
    file_name = exc_tb.tb_frame.f_code.co_filename
    file_name,exc_tb.tb_lineno,str(error)


class ExceptionCustom(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message,error_detail)
        self.error_message = erro_message_detail(error_message,error_detail)
    
    def __str__(self):
        return self.error_message