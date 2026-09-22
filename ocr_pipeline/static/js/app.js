const STATUS_PILL = {
  completed: 'pill-success',
  failed: 'pill-error',
  processing: 'pill-active',
  queued: 'pill-warn',
};
const PAGE_STATUS = {
  pass: 'page-pass',
  warn: 'page-warn',
  fail: 'page-fail',
  empty: 'page-empty',
};

function emptyHFModal() {
  return { open: false, runId: null, repoId: '', private: false, token: '', submitting: false, result: null };
}

function emptyInspector() {
  return { documentId: null, document: null, pageNum: 1, mode: 'txt', text: '' };
}

function opencrApp() {
  return {
    activeView: 'documents',
    version: '',
    healthStatus: 'checking...',
    healthClass: '',
    currentModel: 'deepseek-ai/DeepSeek-OCR-2',
    metrics: {},

    auth: { enabled: false, authenticated: false, user: null },

    extractionProfiles: [
      {
        id: 'latinized_ottoman_careful',
        role: 'base profile',
        status: 'default',
        engine: 'vLLM',
        model: 'deepseek-ai/DeepSeek-OCR-2',
        dpi: '300 / 400',
        prompt: 'Free OCR or grounded markdown',
        crop: 'benchmark on/off',
        cleanup: 'conservative',
      },
      {
        id: 'ottoman_arabic_layout',
        role: 'candidate',
        status: 'benchmark',
        engine: 'vLLM + layout detector',
        model: 'deepseek-ai/DeepSeek-OCR-2',
        dpi: '400',
        prompt: 'grounded markdown',
        crop: 'on',
        cleanup: 'conservative',
      },
      {
        id: 'tesseract_turkish_baseline',
        role: 'baseline',
        status: 'baseline',
        engine: 'Tesseract LSTM',
        model: 'tur',
        dpi: '300',
        prompt: 'n/a',
        crop: 'off',
        cleanup: 'minimal',
      },
    ],

    benchmarkRows: [
      {
        documentType: 'Latinized Ottoman',
        profile: 'latinized_ottoman_careful',
        cer: 'pending',
        wer: 'pending',
        decision: 'measure',
        best: false,
      },
      {
        documentType: 'Ottoman Arabic print',
        profile: 'ottoman_arabic_layout',
        cer: 'pending',
        wer: 'pending',
        decision: 'measure',
        best: false,
      },
      {
        documentType: 'Modern Turkish print',
        profile: 'tesseract_turkish_baseline',
        cer: 'pending',
        wer: 'pending',
        decision: 'baseline',
        best: false,
      },
    ],

    runs: [],
    selectedRunId: null,
    selectedRun: null,
    selectedRunDocumentIds: [],

    documents: [],
    selectedDocumentIds: [],
    selectedDocumentId: null,
    documentDraft: {},
    documentSearch: '',
    documentGroupFilter: '',
    bulkGroupPath: '',
    bulkGrouping: false,
    savingDocument: false,

    inputFiles: [],
    selectedPaths: [],
    intakeOptions: { stripRefs: false, exportParquet: true, name: '' },
    creating: false,
    dragOver: false,
    uploadProgress: null,

    inspector: emptyInspector(),
    hfModal: emptyHFModal(),
    toasts: [],

    _stream: new RunStream(),
    _runPollTimer: null,

    async init() {
      await Promise.all([
        this.refreshHealth(),
        this.refreshMetrics(),
        this.refreshInputFiles(),
        this.refreshDocuments(),
        this.refreshRuns(),
        this.refreshAuth(),
      ]);
      setInterval(() => this.refreshHealth(), 15000);
      setInterval(() => this.refreshMetrics(), 10000);
    },

    async refreshAuth() {
      try { this.auth = await API.authMe(); } catch { this.auth = { enabled: false, authenticated: false, user: null }; }
    },

    signIn() { window.location.href = '/api/auth/login'; },

    async signOut() {
      try {
        await API.authLogout();
        await this.refreshAuth();
        this.toast('Signed out', 'info');
      } catch (e) {
        this.toast(`Sign out failed: ${e.message}`, 'error');
      }
    },

    get canPublish() {
      // OAuth disabled → publish is always available (paste-token mode).
      // OAuth enabled  → publish requires a signed-in user.
      return !this.auth.enabled || this.auth.authenticated;
    },

    async refreshHealth() {
      try {
        const data = await API.health();
        this.version = data.pipeline_version || '';
        this.healthStatus = data.model_status || data.status;
        this.currentModel = data.model_name || this.currentModel;
        this.healthClass = data.status === 'ready' ? 'ready' : 'waiting';
      } catch {
        this.healthStatus = 'offline';
        this.healthClass = 'error';
      }
    },

    async refreshMetrics() {
      try { this.metrics = await API.metricsSummary(); } catch {}
    },

    async refreshInputFiles() {
      try { this.inputFiles = await API.listInputFiles(); }
      catch (e) { this.toast(`Failed to load inputs: ${e.message}`, 'error'); }
    },

    async refreshDocuments() {
      try {
        this.documents = await API.listDocuments();
        if (!this.selectedDocumentId && this.documents.length > 0) this.selectDocument(this.documents[0].id);
      } catch (e) {
        this.toast(`Failed to load documents: ${e.message}`, 'error');
      }
    },

    async refreshRuns() {
      try { this.runs = await API.listRuns(50); }
      catch (e) { this.toast(`Failed to load runs: ${e.message}`, 'error'); }
    },

    async refreshSelectedRun() {
      if (!this.selectedRunId) return;
      try {
        this.selectedRun = await API.getRun(this.selectedRunId);
        if (!['queued', 'processing'].includes(this.selectedRun.status)) {
          this.stopRunPolling();
        }
      }
      catch (e) { this.toast(`Failed to refresh run: ${e.message}`, 'error'); }
    },

    async selectRun(runId) {
      this._stream.disconnect();
      this.stopRunPolling();
      if (!runId) {
        this.activeView = 'documents';
        this.selectedRunId = null;
        this.selectedRun = null;
        this.selectedRunDocumentIds = [];
        this.inspector = emptyInspector();
        return;
      }
      this.activeView = 'runs';
      this.selectedRunId = runId;
      try {
        this.selectedRun = await API.getRun(runId);
        this.selectedRunDocumentIds = [];
        const firstCompleted = (this.selectedRun.documents || []).find(d => d.status === 'completed');
        if (firstCompleted) await this.openDocument(firstCompleted.document_id);
        else this.inspector = emptyInspector();
        if (['queued', 'processing'].includes(this.selectedRun.status)) {
          this.connectStream(runId);
          this.startRunPolling();
        }
      } catch (e) {
        this.toast(`Failed to load run: ${e.message}`, 'error');
      }
    },

    connectStream(runId) {
      this._stream.connect(runId, async (event) => {
        await this.refreshSelectedRun();
        if (event.type === 'run_complete' || event.type === 'run_failed') {
          await Promise.all([this.refreshSelectedRun(), this.refreshRuns(), this.refreshMetrics()]);
          const completed = event.type === 'run_complete';
          this.toast(`Run ${runId} ${completed ? 'completed' : 'failed'}`, completed ? 'success' : 'error');
        }
        if (event.type === 'run_started') await this.refreshRuns();
      });
    },

    startRunPolling() {
      this.stopRunPolling();
      this._runPollTimer = setInterval(async () => {
        if (!this.selectedRunId) return this.stopRunPolling();
        await Promise.all([this.refreshSelectedRun(), this.refreshRuns(), this.refreshMetrics()]);
      }, 2000);
    },

    stopRunPolling() {
      if (this._runPollTimer) {
        clearInterval(this._runPollTimer);
        this._runPollTimer = null;
      }
    },

    async openDocument(documentId) {
      if (!this.selectedRunId || !documentId) return;
      this.inspector.documentId = documentId;
      try {
        this.inspector.document = await API.getRunDocument(this.selectedRunId, documentId);
        this.inspector.pageNum = 1;
        await this.loadInspectorText();
      } catch (e) {
        this.toast(`Open document failed: ${e.message}`, 'error');
      }
    },

    async loadInspectorText() {
      if (!this.selectedRunId || !this.inspector.documentId) return;
      try {
        this.inspector.text = await API.getDocumentText(
          this.selectedRunId, this.inspector.documentId, this.inspector.mode,
        );
      } catch (e) {
        this.inspector.text = `(${e.message})`;
      }
    },

    setInspectorMode(mode) {
      this.inspector.mode = mode;
      this.loadInspectorText();
    },

    setInspectorPage(num) {
      const total = this.inspector.document?.total_pages || 1;
      this.inspector.pageNum = Math.min(Math.max(1, num), total);
    },

    pageImageUrl(pageNum) {
      if (!this.selectedRunId || !this.inspector.documentId) return '';
      return API.pageImageUrl(this.selectedRunId, this.inspector.documentId, pageNum);
    },

    selectedPageText() {
      const text = this.inspector.text || '';
      if (!text) return '';
      const parts = text.split('\n\f\n');
      if (parts.length <= 1) return text;
      return parts[this.inspector.pageNum - 1] || '';
    },

    currentPageMeta() {
      return (this.inspector.document?.pages || [])
        .find(p => p.page_num === this.inspector.pageNum) || null;
    },

    currentPageQualityFlags() {
      return this.currentPageMeta()?.quality_flags || [];
    },

    currentPageIssues() {
      return this.currentPageMeta()?.validation_issues || [];
    },

    pageStatusFor(pageNum) {
      return (this.inspector.document?.pages || []).find(p => p.page_num === pageNum)?.status || 'pending';
    },

    currentRunDocument() {
      const docs = this.selectedRun?.documents || [];
      if (docs.length === 0) return null;
      return docs.find(d => d.status === 'processing')
        || docs.find(d => ['pending', 'queued'].includes(d.status))
        || docs.find(d => d.status === 'failed')
        || docs[docs.length - 1];
    },

    currentRunDocumentIndex() {
      const docs = this.selectedRun?.documents || [];
      const current = this.currentRunDocument();
      const index = current ? docs.findIndex(d => d.document_id === current.document_id) : -1;
      return index === -1 ? 0 : index + 1;
    },

    runDocumentProgressLabel() {
      const total = this.selectedRun?.documents_total || (this.selectedRun?.documents || []).length || 0;
      return `[${this.currentRunDocumentIndex()}/${total}]`;
    },

    runProgressPercent() {
      return Math.max(0, Math.min(100, Math.round((this.selectedRun?.progress || 0) * 100)));
    },

    runPageProgressLabel() {
      return `${this.selectedRun?.pages_completed || 0}/${this.selectedRun?.pages_total || 0}`;
    },

    runStatsLabel() {
      const docs = this.selectedRun?.documents || [];
      const warn = docs.reduce((sum, d) => sum + (d.pages_warn || 0), 0);
      const fail = docs.reduce((sum, d) => sum + (d.pages_fail || 0), 0);
      return `${this.runProgressPercent()}% · ${warn} warn · ${fail} fail`;
    },

    pageStatusClass(status) { return PAGE_STATUS[status] || 'page-pending'; },
    runStatusClass(status) { return STATUS_PILL[status] || 'pill-muted'; },
    profileStatusClass(status) {
      if (status === 'default') return 'pill-success';
      if (status === 'benchmark') return 'pill-warn';
      return 'pill-muted';
    },

    setActiveView(view) {
      this.activeView = view;
    },

    documentProcessLabel(doc) {
      const status = doc.latest_run_status;
      if (status === 'completed') return 'processed';
      if (['queued', 'processing'].includes(status)) return 'running';
      if (status === 'failed') return 'failed';
      return 'never';
    },

    documentProcessClass(doc) {
      const status = doc.latest_run_status;
      if (status === 'completed') return 'pill-success';
      if (['queued', 'processing'].includes(status)) return 'pill-active';
      if (status === 'failed') return 'pill-error';
      return 'pill-muted';
    },

    toggleRunDocument(documentId) {
      const i = this.selectedRunDocumentIds.indexOf(documentId);
      if (i === -1) this.selectedRunDocumentIds.push(documentId);
      else this.selectedRunDocumentIds.splice(i, 1);
    },

    selectedDocument() {
      return this.documents.find(d => d.id === this.selectedDocumentId) || null;
    },

    availableDocumentGroups() {
      return [...new Set(this.documents.map(d => (d.group_path || '').trim()).filter(Boolean))].sort();
    },

    filteredDocuments() {
      const query = this.documentSearch.trim().toLowerCase();
      return this.documents.filter((doc) => {
        const group = (doc.group_path || '').trim();
        if (this.documentGroupFilter && group !== this.documentGroupFilter) return false;
        if (!query) return true;
        return [
          doc.display_title,
          doc.filename,
          doc.group_path,
          doc.author,
          doc.work,
          doc.book,
          doc.document_date_label,
          doc.language,
          doc.script,
        ].some(value => String(value || '').toLowerCase().includes(query));
      });
    },

    groupedDocuments() {
      const groups = new Map();
      for (const doc of this.filteredDocuments()) {
        const name = (doc.group_path || '').trim() || 'Ungrouped';
        if (!groups.has(name)) groups.set(name, []);
        groups.get(name).push(doc);
      }
      return [...groups.entries()].map(([name, items]) => ({ name, items }));
    },

    selectDocument(documentId) {
      this.selectedDocumentId = documentId;
      const doc = this.selectedDocument();
      this.documentDraft = doc ? { ...doc } : {};
    },

    toggleDocument(documentId) {
      const i = this.selectedDocumentIds.indexOf(documentId);
      if (i === -1) this.selectedDocumentIds.push(documentId);
      else this.selectedDocumentIds.splice(i, 1);
    },

    selectAllDocuments(checked) {
      const visibleIds = this.filteredDocuments().map(d => d.id);
      if (!checked) {
        this.selectedDocumentIds = this.selectedDocumentIds.filter(id => !visibleIds.includes(id));
        return;
      }
      this.selectedDocumentIds = [...new Set([...this.selectedDocumentIds, ...visibleIds])];
    },

    get allDocumentsSelected() {
      const visibleIds = this.filteredDocuments().map(d => d.id);
      return visibleIds.length > 0 && visibleIds.every(id => this.selectedDocumentIds.includes(id));
    },

    selectedDocumentPaths() {
      return this.documents
        .filter(d => this.selectedDocumentIds.includes(d.id))
        .map(d => d.source_path);
    },

    async saveSelectedDocument() {
      if (!this.selectedDocumentId || this.savingDocument) return;
      this.savingDocument = true;
      try {
        await API.updateDocument(this.selectedDocumentId, {
          display_title: this.documentDraft.display_title || null,
          group_path: this.documentDraft.group_path || null,
          author: this.documentDraft.author || null,
          work: this.documentDraft.work || null,
          book: this.documentDraft.book || null,
          document_date_label: this.documentDraft.document_date_label || null,
          document_date_precision: this.documentDraft.document_date_precision || null,
          language: this.documentDraft.language || null,
          script: this.documentDraft.script || null,
          license: this.documentDraft.license || null,
          source_citation: this.documentDraft.source_citation || null,
          notes: this.documentDraft.notes || null,
        });
        await this.refreshDocuments();
        this.selectDocument(this.selectedDocumentId);
        this.toast('Document metadata saved', 'success');
      } catch (e) {
        this.toast(`Metadata save failed: ${e.message}`, 'error');
      } finally {
        this.savingDocument = false;
      }
    },

    async applyBulkGroup() {
      if (this.selectedDocumentIds.length === 0 || this.bulkGrouping) return;
      this.bulkGrouping = true;
      try {
        await API.bulkUpdateDocuments({
          document_ids: this.selectedDocumentIds,
          group_path: this.bulkGroupPath || null,
        });
        await this.refreshDocuments();
        if (this.selectedDocumentId) this.selectDocument(this.selectedDocumentId);
        this.toast('Group updated', 'success');
      } catch (e) {
        this.toast(`Group update failed: ${e.message}`, 'error');
      } finally {
        this.bulkGrouping = false;
      }
    },

    async startDocumentsRun() {
      const paths = this.selectedDocumentPaths();
      if (paths.length === 0) return this.toast('Select documents first', 'error');
      const alreadyProcessed = this.documents.filter(
        d => this.selectedDocumentIds.includes(d.id) && d.latest_run_status === 'completed',
      );
      if (alreadyProcessed.length > 0 && !confirm(`${alreadyProcessed.length} selected document(s) were already processed. Start a new run anyway?`)) {
        return;
      }
      this.selectedPaths = paths;
      await this.startNewRun();
    },

    async startNewRun() {
      if (this.selectedPaths.length === 0 || this.creating) return;
      this.creating = true;
      try {
        const result = await API.createRun(this.selectedPaths, {
          name: this.intakeOptions.name || null,
          stripRefs: this.intakeOptions.stripRefs,
          exportParquet: this.intakeOptions.exportParquet,
        });
        const dedup = result.documents.filter(d => d.deduped).length;
        if (dedup > 0) this.toast(`${dedup} document(s) recognized from prior runs`, 'info');
        this.toast(`Run ${result.run_id} queued`, 'success');
        this.selectedPaths = [];
        this.selectedDocumentIds = [];
        await this.refreshRuns();
        await this.selectRun(result.run_id);
      } catch (e) {
        this.toast(`Failed to start run: ${e.message}`, 'error');
      } finally {
        this.creating = false;
      }
    },

    async deleteSelectedRun() {
      if (!this.selectedRunId) return;
      if (!confirm(`Delete run ${this.selectedRunId}? Artifact files remain on disk.`)) return;
      try {
        await API.deleteRun(this.selectedRunId);
        this.toast(`Run ${this.selectedRunId} deleted`, 'info');
        await this.selectRun(null);
        await this.refreshRuns();
      } catch (e) {
        this.toast(`Delete failed: ${e.message}`, 'error');
      }
    },

    async retryRun() {
      if (!this.selectedRunId || this.creating) return;
      this.creating = true;
      try {
        const result = await API.retryRun(this.selectedRunId);
        this.toast(`Retry run ${result.run_id} queued`, 'success');
        await this.refreshRuns();
        await this.selectRun(result.run_id);
      } catch (e) {
        this.toast(`Retry failed: ${e.message}`, 'error');
      } finally {
        this.creating = false;
      }
    },

    toggleSelected(path) {
      const i = this.selectedPaths.indexOf(path);
      if (i === -1) this.selectedPaths.push(path); else this.selectedPaths.splice(i, 1);
    },

    selectAllInputs(checked) {
      this.selectedPaths = checked ? this.inputFiles.map(f => f.path) : [];
    },

    get allInputsSelected() {
      return this.inputFiles.length > 0 && this.selectedPaths.length === this.inputFiles.length;
    },

    async handleDrop(event) {
      this.dragOver = false;
      const files = Array.from(event.dataTransfer.files).filter(f => /\.(pdf|epub)$/i.test(f.name));
      if (files.length === 0) return this.toast('Only PDF and EPUB files are accepted', 'error');
      await this.uploadFiles(files);
    },

    async handleFileSelect(event) {
      const files = Array.from(event.target.files || []);
      if (files.length > 0) await this.uploadFiles(files);
      event.target.value = '';
    },

    async uploadFiles(files) {
      let count = 0;
      for (const file of files) {
        try {
          this.uploadProgress = 0;
          await API.upload(file, p => { this.uploadProgress = p; });
          count += 1;
        } catch (e) {
          this.toast(`Upload failed: ${file.name} — ${e.message}`, 'error');
        }
      }
      this.uploadProgress = null;
      if (count > 0) {
        this.toast(`Uploaded ${count} file(s)`, 'success');
        await Promise.all([this.refreshInputFiles(), this.refreshDocuments()]);
      }
    },

    downloadArtifact(documentId, artifact) {
      this._download(API.artifactDownloadUrl(this.selectedRunId, documentId, artifact));
    },

    downloadDataset() {
      if (this.selectedRunId) this._download(API.datasetDownloadUrl(this.selectedRunId));
    },

    downloadOCRPairs() {
      if (this.selectedRunId) {
        this._download(API.ocrPairsDownloadUrl(
          this.selectedRunId,
          { documentIds: this.selectedRunDocumentIds },
        ));
      }
    },

    downloadTextBundle() {
      if (this.selectedRunId) {
        this._download(API.textBundleDownloadUrl(
          this.selectedRunId,
          { documentIds: this.selectedRunDocumentIds },
        ));
      }
    },

    openHFModal() {
      if (!this.selectedRunId) return;
      if (!this.canPublish) {
        this.toast('Sign in with HuggingFace to publish.', 'error');
        return;
      }
      const defaultRepo = this.auth.user?.name
        ? `${this.auth.user.name}/${this.selectedRun?.name || `opencr-${this.selectedRunId}`}`
        : '';
      this.hfModal = { ...emptyHFModal(), open: true, runId: this.selectedRunId, repoId: defaultRepo };
    },

    closeHFModal() { this.hfModal.open = false; },

    async submitHFPublish() {
      if (!this.hfModal.repoId) return this.toast('Repo id required (e.g. user/dataset)', 'error');
      this.hfModal.submitting = true;
      try {
        this.hfModal.result = await API.publishToHF(this.hfModal.runId, {
          repo_id: this.hfModal.repoId,
          private: this.hfModal.private,
          token: this.hfModal.token || null,
        });
        this.toast('Published to HuggingFace', 'success');
      } catch (e) {
        this.toast(`Publish failed: ${e.message}`, 'error');
      } finally {
        this.hfModal.submitting = false;
      }
    },

    _download(url) {
      const a = document.createElement('a');
      a.href = url; a.target = '_blank'; a.rel = 'noopener';
      document.body.appendChild(a); a.click(); a.remove();
    },

    formatSize(bytes) {
      if (!bytes) return '0 B';
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / 1048576).toFixed(1)} MB`;
    },

    formatMetric(value, digits = 1) { return Number(value || 0).toFixed(digits); },

    formatTimestamp(iso) {
      if (!iso) return '—';
      try { return new Date(iso).toLocaleString(); } catch { return iso; }
    },

    toast(message, type = 'info') {
      const id = Math.random().toString(36).slice(2);
      this.toasts.push({ id, message, type });
      setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, 4000);
    },
  };
}
