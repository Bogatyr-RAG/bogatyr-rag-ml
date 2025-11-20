import os
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileStatus(Enum):
    """Класс для представления статуса файла"""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"

@dataclass
class FileInfo:
    """Класс для описания метаданных файла"""
    file_path: str
    stable_id: str
    content_hash: str
    file_size: int
    last_modified: float
    file_type: str

@dataclass
class ScanResult:
    """Класс содержащий классифицированные списки файлов"""
    added_files: List[FileInfo]
    modified_files: List[FileInfo]
    deleted_files: List[FileInfo]
    unchanged_files: List[FileInfo]
    total_files: int

class FileSystemScanner:
    """
    Компонент сканирования и отслеживания состояния файловой системы
    для системы CompactRAG
    """
    
    def __init__(self, root_path: str, state_file: str = "filesystem_state.json"):
        self.root_path = Path(root_path).resolve()
        self.state_file = Path(state_file)
        self.supported_formats = {
            '.pdf', '.docx', '.xlsx', '.pptx', 
            '.jpg', '.jpeg', '.png', '.txt', '.md', '.html'
        }
        
        # Кеш предыдущего состояния
        self.previous_state: Dict[str, FileInfo] = {}
        self.load_previous_state()
    
    def load_previous_state(self) -> None:
        """Загружает предыдущее состояние файловой системы"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    self.previous_state = {
                        file_path: FileInfo(**file_info) 
                        for file_path, file_info in state_data.items()
                    }
                logger.info(f"Загружено состояние {len(self.previous_state)} файлов")
            else:
                logger.info("Предыдущее состояние не найдено, будет выполнено полное сканирование")
        except Exception as e:
            logger.error(f"Ошибка при загрузке состояния: {e}")
            self.previous_state = {}
    
    def save_current_state(self, current_state: Dict[str, FileInfo]) -> None:
        """Сохраняет текущее состояние файловой системы"""
        try:
            state_data = {
                file_path: asdict(file_info) 
                for file_path, file_info in current_state.items()
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Сохранено состояние {len(current_state)} файлов")
        except Exception as e:
            logger.error(f"Ошибка при сохранении состояния: {e}")
    
    def calculate_stable_id(self, file_path: Path) -> str:
        """
        Вычисляет стабильный ID для файла на основе относительного пути
        """
        try:
            relative_path = file_path.relative_to(self.root_path)
            return str(relative_path).replace('\\', '/')  # Унифицируем разделители
        except ValueError:
            return str(file_path)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Вычисляет хэш содержимого файла
        """
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                # Читаем файл блоками для обработки больших файлов
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Не удалось вычислить хэш для {file_path}: {e}")
            return ""
    
    def get_file_info(self, file_path: Path) -> Optional[FileInfo]:
        """
        Собирает информацию о файле
        """
        try:
            if not file_path.is_file():
                return None
            
            # Пропускаем неподдерживаемые форматы
            if file_path.suffix.lower() not in self.supported_formats:
                return None
            
            stat = file_path.stat()
            
            return FileInfo(
                file_path=str(file_path),
                stable_id=self.calculate_stable_id(file_path),
                content_hash=self.calculate_file_hash(file_path),
                file_size=stat.st_size,
                last_modified=stat.st_mtime,
                file_type=file_path.suffix.lower()
            )
        except Exception as e:
            logger.warning(f"Ошибка при обработке файла {file_path}: {e}")
            return None
    
    def scan_directory(self) -> Dict[str, FileInfo]:
        """
        Рекурсивно сканирует директорию и собирает информацию о файлах
        """
        current_state = {}
        total_files = 0
        processed_files = 0
        
        logger.info(f"Начато сканирование директории: {self.root_path}")
        
        # Рекурсивный обход файловой системы
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file():
                total_files += 1
                file_info = self.get_file_info(file_path)
                if file_info:
                    current_state[file_info.stable_id] = file_info
                    processed_files += 1
        
        logger.info(f"Сканирование завершено. Обработано {processed_files}/{total_files} файлов")
        return current_state
    
    def compare_states(self, current_state: Dict[str, FileInfo]) -> ScanResult:
        """
        Сравнивает текущее состояние с предыдущим и определяет изменения
        """
        added_files = []
        modified_files = []
        deleted_files = []
        unchanged_files = []
        
        current_ids = set(current_state.keys())
        previous_ids = set(self.previous_state.keys())
        
        # Новые файлы
        for file_id in current_ids - previous_ids:
            added_files.append(current_state[file_id])
        
        # Удаленные файлы
        for file_id in previous_ids - current_ids:
            deleted_files.append(self.previous_state[file_id])
        
        # Проверка измененных файлов
        for file_id in current_ids & previous_ids:
            current_file = current_state[file_id]
            previous_file = self.previous_state[file_id]
            
            if (current_file.content_hash != previous_file.content_hash or
                current_file.last_modified != previous_file.last_modified):
                modified_files.append(current_file)
            else:
                unchanged_files.append(current_file)
        
        return ScanResult(
            added_files = added_files,
            modified_files = modified_files,
            deleted_files = deleted_files,
            unchanged_files = unchanged_files,
            total_files = len(current_state)
        )
    
    def scan(self) -> ScanResult:
        """
        Основной метод сканирования с определением изменений
        """
        # Сканируем текущее состояние
        current_state = self.scan_directory()
        
        # Сравниваем с предыдущим состоянием
        scan_result = self.compare_states(current_state)
        
        # Сохраняем новое состояние
        self.save_current_state(current_state)
        self.previous_state = current_state
        
        # Логируем результаты
        self._log_scan_results(scan_result)
        
        return scan_result
    
    def _log_scan_results(self, scan_result: ScanResult) -> None:
        """Логирует результаты сканирования"""
        logger.info("=== РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ ===")
        logger.info(f"Всего файлов в системе: {scan_result.total_files}")
        logger.info(f"Новых файлов: {len(scan_result.added_files)}")
        logger.info(f"Измененных файлов: {len(scan_result.modified_files)}")
        logger.info(f"Удаленных файлов: {len(scan_result.deleted_files)}")
        logger.info(f"Неизмененных файлов: {len(scan_result.unchanged_files)}")
        
        if scan_result.added_files:
            logger.info("Новые файлы:")
            for file_info in scan_result.added_files[:5]: # Показываем первые 5
                logger.info(f"  + {file_info.stable_id}")
        
        if scan_result.modified_files:
            logger.info("Измененные файлы:")
            for file_info in scan_result.modified_files[:5]: # Показываем первые 5
                logger.info(f"  ~ {file_info.stable_id}")
        
        if scan_result.deleted_files:
            logger.info("Удаленные файлы:")
            for file_info in scan_result.deleted_files[:5]: # Показываем первые 5
                logger.info(f"  - {file_info.stable_id}")

# Пример использования
def main():
    # Инициализация сканера
    scanner = FileSystemScanner(
        root_path="D:\\Root\\Directory_for_fefu\\4-курс\\5_семестр — копия",  # Путь к корневой директории с документами (default_value = "./documents")
        state_file="D:\\Root\\Directory_for_projects\\Акселератор\\CompactRAG\\scanner_state.json" # Путь к файлу состояния (default_value = "./filesystem_state.json")
    )
    
    # Выполнение сканирования
    scan_result = scanner.scan()
    
    # Возвращаем результат для дальнейшей обработки в CompactRAG
    return scan_result

# Дополнительные утилиты для работы со сканером
class ScannerUtils:
    @staticmethod
    def get_files_for_processing(scan_result: ScanResult) -> List[FileInfo]:
        """Возвращает список файлов, требующих обработки (новые + измененные)"""
        return scan_result.added_files + scan_result.modified_files
    
    @staticmethod
    def get_file_ids_for_deletion(scan_result: ScanResult) -> List[str]:
        """Возвращает список ID файлов для удаления из индекса"""
        return [file_info.stable_id for file_info in scan_result.deleted_files]
    
    @staticmethod
    def format_results_for_commit(scan_result: ScanResult) -> Dict:
        """Форматирует результаты для commit-механизма"""
        return {
            "commit_timestamp": time.time(),
            "files_to_process": [
                {"stable_id": file.stable_id, "file_path": file.file_path} 
                for file in ScannerUtils.get_files_for_processing(scan_result)
            ],
            "files_to_delete": ScannerUtils.get_file_ids_for_deletion(scan_result),
            "summary": {
                "total_processed": len(scan_result.added_files) + len(scan_result.modified_files),
                "total_deleted": len(scan_result.deleted_files),
                "total_unchanged": len(scan_result.unchanged_files)
            }
        }

if __name__ == "__main__":
    # Запуск примера
    result = main()
    
    # Форматирование результатов для commit-процесса
    commit_data = ScannerUtils.format_results_for_commit(result)
    print("\nДанные для commit-процесса:")
    print(json.dumps(commit_data, indent=2, ensure_ascii=False))
    print()