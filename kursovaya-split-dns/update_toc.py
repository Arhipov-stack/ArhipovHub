"""Заполняет поле «Содержание» номерами страниц через LibreOffice (UNO).

docx-js вставляет оглавление как пустое поле, которое Word обновляет
только по запросу. Скрипт открывает документ в LibreOffice, обновляет
все указатели и сохраняет результат обратно в .docx.

Запуск: python3 update_toc.py Kursovaya_Split-DNS_BIND9_AD.docx
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue


def prop(name, value):
    p = PropertyValue()
    p.Name, p.Value = name, value
    return p


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(path):
    src = Path(path).resolve()
    port = free_port()
    profile = tempfile.mkdtemp(prefix="lo_profile_")
    env = dict(os.environ, SAL_USE_VCLPLUGIN="svp")
    proc = subprocess.Popen([
        "soffice", "--headless", "--norestore", "--nologo",
        f"-env:UserInstallation={Path(profile).as_uri()}",
        f"--accept=socket,host=127.0.0.1,port={port};urp;",
    ], env=env)
    try:
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local)
        for _ in range(60):
            try:
                ctx = resolver.resolve(
                    f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("LibreOffice не запустился")
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL(src.as_uri(), "_blank", 0, (prop("Hidden", True),))
        indexes = doc.getDocumentIndexes()
        for i in range(indexes.getCount()):
            indexes.getByIndex(i).update()
        doc.refresh()
        # второй проход: после вставки строк оглавления страницы могли сдвинуться
        for i in range(indexes.getCount()):
            indexes.getByIndex(i).update()
        count = indexes.getCount()
        doc.storeToURL(src.as_uri(), (prop("FilterName", "MS Word 2007 XML"),))
        doc.close(True)
        print(f"Оглавление обновлено: {src.name} (указателей: {count})")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main(sys.argv[1])
