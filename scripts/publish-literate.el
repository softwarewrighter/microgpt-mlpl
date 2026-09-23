;;; publish-literate.el --- Evaluate a docs/literate/*.org file and export HTML -*- lexical-binding: t; -*-

;; Run by scripts/publish-literate.sh:
;;   emacs -Q --batch -l scripts/publish-literate.el ORG-FILE MLPL-ELISP-DIR MLPL-COMMAND
;; Loads sw-mlpl's Org-babel support, evaluates every block (baking
;; #+RESULTS in), saves the .org, and exports HTML beside it.

(let* ((args command-line-args-left)
       (org-file (expand-file-name (nth 0 args)))
       (elisp-dir (expand-file-name (nth 1 args)))
       (mlpl-command (nth 2 args)))
  (setq command-line-args-left nil)
  (load (expand-file-name "mlpl-all.el" elisp-dir) nil t)
  ;; htmlize (NonGNU ELPA) colors the exported source blocks. -Q skips
  ;; package activation, so put an installed copy on the load path.
  (dolist (dir (file-expand-wildcards (expand-file-name "~/.emacs.d/elpa/htmlize-*")))
    (add-to-list 'load-path dir))
  (if (require 'htmlize nil t)
      ;; Batch Emacs has no display, so faces carry no colors: emit
      ;; class names (org-keyword, org-string, ...) and let the
      ;; stylesheet in docs/microgpt.org color them.
      (setq org-html-htmlize-output-type 'css
            org-html-htmlize-font-prefix "org-")
    (message "publish-literate: htmlize not found; source blocks will be plain"))
  (require 'ob)
  (require 'org)
  (require 'ox-html)
  (setq org-confirm-babel-evaluate nil
        make-backup-files nil
        enable-local-variables nil      ; the command comes from the caller
        org-html-validation-link nil)
  (org-babel-do-load-languages 'org-babel-load-languages
                               '((mlpl . t) (emacs-lisp . t)))
  (setq org-babel-mlpl-command mlpl-command)
  (with-current-buffer (find-file-noselect org-file)
    (org-babel-mlpl-reset-session)
    (org-babel-remove-result-one-or-many t)
    (org-babel-execute-buffer)
    (save-buffer)
    (org-html-export-to-html)
    (org-babel-mlpl-reset-session)))
