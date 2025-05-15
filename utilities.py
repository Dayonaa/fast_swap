import os
import shutil

class Utilities:

    @staticmethod
    def make_dir(path: str):
        """Membuat direktori jika belum ada."""
        try:
            os.makedirs(path, exist_ok=True)
            print(f"📁 Direktori dibuat: {path}")
        except Exception as e:
            print(f"❌ Gagal membuat direktori {path}: {e}")

    @staticmethod
    def remove_dir(path: str):
        """Menghapus direktori beserta isinya."""
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"🗑️ Direktori dihapus: {path}")
            else:
                print(f"⚠️ Direktori tidak ditemukan: {path}")
        except Exception as e:
            print(f"❌ Gagal menghapus direktori {path}: {e}")

    @staticmethod
    def remove_file(path: str):
        """Menghapus file tunggal."""
        try:
            if os.path.isfile(path):
                os.remove(path)
                print(f"🗑️ File dihapus: {path}")
            else:
                print(f"⚠️ File tidak ditemukan: {path}")
        except Exception as e:
            print(f"❌ Gagal menghapus file {path}: {e}")

    @staticmethod
    def path_exists(path: str) -> bool:
        """Cek apakah path ada."""
        return os.path.exists(path)

    @staticmethod
    def list_dir(path: str):
        """List semua isi direktori."""
        try:
            if os.path.isdir(path):
                contents = os.listdir(path)
                print(f"📂 Isi direktori '{path}': {contents}")
                return contents
            else:
                print(f"⚠️ {path} bukan direktori.")
                return []
        except Exception as e:
            print(f"❌ Gagal membaca direktori {path}: {e}")
            return []

    @staticmethod
    def get_cwd() -> str:
        """Mendapatkan current working directory."""
        return os.getcwd()
